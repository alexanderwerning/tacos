import warnings

from tacos.datasets.audioset import AudioSetStrong

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

import os
from typing import Union, List, Mapping
import torch
import wandb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from aac_datasets import Clotho
from tacos.datasets.aac_wrapper import AACWrapper
from torch.utils.data import DataLoader, WeightedRandomSampler
from tacos.datasets.download_datasets import download_clotho
from tacos.datasets.tacos import Tacos
from tacos.datasets.utils import exclude_broken_files, exclude_forbidden_files
from tacos.datasets.batch_collate import CustomCollate

from tacos.retrieval_module import AudioRetrievalModel
from tacos.supervised_module import SupervisedModel

from collections import Counter

def train(
        model: AudioRetrievalModel,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
):
    """
    Trains the AudioRetrievalModel using provided datasets, logger, and configuration arguments.

    Args:
        model (tacos.retrieval_module.AudioRetrievalModel): The model to be trained.
        train_ds (torch.utils.data.Dataset): The training dataset.
        val_ds (torch.utils.data.Dataset): The validation dataset.
        logger (Union[None, WandbLogger]): The logger for tracking training metrics.
        args (dict): A dictionary of configuration arguments for training.

    Returns:
        tacos.retrieval_module.AudioRetrievalModel: The trained model.
    """
    # get a unique experiment name for name of checkpoint
    if wandb.run is not None:
        experiment_name = wandb.run.name or wandb.run.id  # Use name if available, else use ID
    else:
        experiment_name = "experiment_" + wandb.util.generate_id()  # Random unique ID fallback

    # create path for the model checkpoints
    checkpoint_dir = os.path.join(args["checkpoints_path"], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch}",
        save_top_k=1,
        monitor="val/pauroc" if args["val_on_audioset"] else "val/mAP@10",
        mode="max",
        save_last=True
    )

    # trainer
    trainer = pl.Trainer(
        devices=args['devices'],
        logger=logger if wandb.run else None,
        enable_checkpointing=args['checkpoint'],
        callbacks=[checkpoint_callback] if args['checkpoint'] else [],
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False
    )


    train_dl = DataLoader(
        train_ds, batch_size=args['batch_size'], num_workers=args['n_workers'], shuffle=True, drop_last=True,
        persistent_workers=True, collate_fn=CustomCollate(), pin_memory=True
    )

    ### train on training set; monitor performance on val
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=DataLoader(
            val_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate(), pin_memory=True
        ),
        ckpt_path=args['resume_ckpt_path'] # should be none unless training is resumed
    )

    return model

def test(
        model: AudioRetrievalModel,
        test_ds: torch.utils.data.Dataset,
        logger: Union[None, WandbLogger],
        args: dict
) -> List[Mapping[str, float]]:
    """
    Tests the trained AudioRetrievalModel on a given test dataset.

    Args:
        model (tacos.retrieval_module.AudioRetrievalModel): The trained model to be evaluated.
        test_ds (torch.utils.data.Dataset): The test dataset.
        logger (Union[None, WandbLogger]): The logger for tracking test metrics.
        args (dict): A dictionary of configuration arguments for testing.

    Returns:
        dict: The result of the model evaluation on the test dataset.
    """
    trainer = pl.Trainer(
        devices=args['devices'],
        enable_checkpointing=False,
        logger=logger if wandb.run else None,
        callbacks=None,
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False
    )

    ### test on the eval set
    result = trainer.test(
        model,
        DataLoader(
            test_ds, batch_size=args['batch_size_eval'], num_workers=args['n_workers'], shuffle=False, drop_last=False,
            persistent_workers=True, collate_fn=CustomCollate(), pin_memory=True
        )
    )

    return result


def get_args() -> dict:
    """
    Parses command-line arguments for configuring the training and testing process.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Argument parser for training configuration.")

    parser.add_argument('--devices', type=str, default='auto', help='Device selection (e.g., auto, cpu, cuda, etc.)')
    parser.add_argument('--n_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction, help='Compile the model if GPU version >= 7.')
    parser.add_argument('--logging', default=True, action=argparse.BooleanOptionalAction, help='Log metrics in wandb or not.')
    parser.add_argument('--preload_audios', default=False, action=argparse.BooleanOptionalAction, help='Store audios in memory.')
    parser.add_argument('--checkpoint', default=False, action=argparse.BooleanOptionalAction, help='Save checkpoints.')

    # Parameter initialization & resume training
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='Path to checkpoint used as a weight initialization for training.')

    # Training parameters
    parser.add_argument('--seed', type=int, default=-1, help='Random seed of experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--batch_size_eval', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--rampdown_epochs', type=int, default=15, help='Number of ramp-down epochs')
    parser.add_argument('--max_lr', type=float, default=2e-5, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--initial_tau', type=float, default=0.05, help='Initial tau value')
    parser.add_argument('--tau_trainable', default=False, action=argparse.BooleanOptionalAction, help='Temperature parameter is trainable or not.')
    parser.add_argument('--val_on_audioset', default=False, action=argparse.BooleanOptionalAction, help='Validate on AudioSet or not.')

    # RoBERTa parameters
    parser.add_argument('--roberta_base', default=True, action=argparse.BooleanOptionalAction,  help='Use Roberta base or large.')

    # use additional data sets...
    parser.add_argument('--clotho', default=True, action=argparse.BooleanOptionalAction,
                        help='Include Clotho in the training or not.')
    parser.add_argument('--tacos', default=False, action=argparse.BooleanOptionalAction,
                        help='Include Tacos in the training or not.')
    parser.add_argument('--audiocaps', default=False, action=argparse.BooleanOptionalAction,
                        help='Include AudioCaps in the training or not.')
    # Paths
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset; dataset will be downloaded into this folder.')
    parser.add_argument('--cache_path', type=str, default='.', help='Path to where h5py files will be stores.')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='Path to save checkpoints to.')

    # run training / test
    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction, help='Run training or not.')
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction, help='Run testing or not.')
    parser.add_argument('--test_on_audioset', default=False, action=argparse.BooleanOptionalAction, help='Run testing on audioset strong.')
    parser.add_argument('--test_on_audioset_full', default=False, action=argparse.BooleanOptionalAction,
                        help='Run testing on audioset strong full.')
    parser.add_argument('--weak_weight', type=float, default=1, help='')
    parser.add_argument('--strong_weight', type=float, default=0, help='')
    parser.add_argument('--strong_tau', type=float, default=1.0, help='')

    parser.add_argument('--audio_embedding_checkpoint', type=str, default='ASIT_strong_1', help='Checkpoint for audio embedding model.')
    parser.add_argument('--notes', type=str, default='', help='Notes to identify the run.')

    parser.add_argument('--freeze_text_encoder', default=False, action=argparse.BooleanOptionalAction,
                        help='Freeze text encoder for stong experiments.')

    parser.add_argument('--evaluate_supervised', default=False, action=argparse.BooleanOptionalAction,
                        help='Evaluate on the supervised model.')

    parser.add_argument('--median_filter', type=int, default=9, help='Median Filter')

    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    """
    Entry point for training and testing the model.
    - Downloads datasets if necessary.
    - Initializes logging and model.
    - Runs training and/or testing based on arguments.
    """
    args = get_args()

    os.makedirs(args["data_path"], exist_ok=True)
    # download data sets; will be ignored if exists
    # ClothoV2.1
    download_clotho(args["data_path"])
    # AudioCAps
    if args['audiocaps']:
        download_audiocaps(args["data_path"])

    import secrets
    # set a seed to make experiments reproducible
    if args['seed'] > 0:
        seed_everything(args['seed'], workers=True)
    else:
        args['seed'] = secrets.randbelow(2 ** 32)  # generates a secure random seed
        seed_everything(args['seed'], workers=True)

    # initialize wandb, i.e., the logging framework
    if args['logging']:
        wandb.init(project="tacos")
        logger = WandbLogger(notes=args["notes"])
    else:
        logger = None

    if args['evaluate_supervised']:
        model = SupervisedModel(**args)
    # initialize the model
    elif args['load_ckpt_path']:
        ckpt = args['checkpoint']
        del args['checkpoint']
        model = AudioRetrievalModel.load_from_checkpoint(args['load_ckpt_path'], **args, strict=False)
        args['checkpoint'] = ckpt
        new_tau = torch.tensor([args['initial_tau']], device=model.tau.device)  # or whatever value you want
        model.tau = torch.nn.Parameter(new_tau, requires_grad=args['tau_trainable'])
    else:
        model = AudioRetrievalModel(**args)

    from tacos.datasets.utils import CacheDataSet
    # train
    if args['train'] and not args['evaluate_supervised']:
        # get training ad validation data sets; add the resampling transformation
        train_ds = []

        # Clotho
        if args['clotho']:
            c = AACWrapper(Clotho(subset="dev", root=args["data_path"], flat_captions=True))
            if args['preload_audios']: c.preload_audios(n_workers=args["n_workers"]+1, show_progress=True, cache_path=args["cache_path"])
            train_ds.append(c)

        # Tacos
        if args['tacos']:
            tacos = Tacos(args["data_path"])
            if args['preload_audios']: tacos.preload_audios(n_workers=args["n_workers"]+1, show_progress=True, cache_path=args["cache_path"])
            train_ds.append(tacos)

        # concatenate
        if len(train_ds) > 1:
            train_ds = torch.utils.data.ConcatDataset(train_ds)
        else:
            train_ds = train_ds[0]

        # load val set
        val_ds = AACWrapper(Clotho(subset="val", root=args["data_path"], flat_captions=True))
        if args['preload_audios']: val_ds.preload_audios(n_workers=args["n_workers"]+1, show_progress=True, cache_path=args["cache_path"])

        model = train(model, train_ds, val_ds, logger, args)

    # test
    if args['test']:

        # load test
        if args['test_on_audioset'] or args['evaluate_supervised']:
            from tacos.datasets.audioset import AudioSetStrong

            test_ds = AudioSetStrong(reduce_classes=not args['test_on_audioset_full'], to_sentences=not args['evaluate_supervised'])
        else:
            test_ds = AACWrapper(Clotho(subset="eval", root=args["data_path"], flat_captions=True))

        if args['preload_audios']: test_ds.preload_audios(n_workers=args["n_workers"] + 1, show_progress=True, cache_path=args["cache_path"])
        results = test(model, test_ds, logger, args)
