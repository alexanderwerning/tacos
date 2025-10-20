import copy
import math
import os
from typing import Any
import math

import numpy as np
import torch
import pytorch_lightning as pl

from tacos.models.sed import ASITSEDWrapper

import torch

class SupervisedModel(pl.LightningModule):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__()
        self.save_hyperparameters(kwargs)

        # audio encoder
        self.audio_embedding_model = ASITSEDWrapper(checkpoint=kwargs['audio_embedding_checkpoint'], n_classes_strong=447, rnn_layers=0, seq_len=250)

        self.validation_outputs = []

        self.kwargs = kwargs

        self.compile_model()

    def compile_model(self):
        """Apply torch.compile() if GPU is recent"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()  # Get current GPU device
            properties = torch.cuda.get_device_properties(device)
            if properties.major >= 7 and self.kwargs['compile'] == True:
                print("Compiling Models")
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)

    def forward(self, batch) -> Any:
        # embed audio & text
        audio_embeddings = self.forward_audio(batch)

        return audio_embeddings


    def forward_audio(self, batch):
        audio = batch['audio'][:, :, :320000]
        audio_embedding_sequence = self.audio_embedding_model(
            audio.mean(1),
            duration=torch.tensor([ min([l, 10]) for l in batch["duration"]]) / 10
        ).permute(0,2,1) # forward

        batch['audio_embedding'] = audio_embedding_sequence
        return audio_embedding_sequence


    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        self.forward(batch)
        assert batch['dataset'][0] == 'audioset_strong'
        args = {
            'audio_embeddings': copy.deepcopy(batch['audio_embedding'].detach().cpu()),
            'path': batch['fname'],
            'onsets': batch['onsets'],
            'captions': batch['captions_strong'],
            'offsets': batch['offsets'],
            'duration': batch['duration'],
            'dataset': batch['dataset'],
            'fname': batch['fname']
        }

        self.validation_outputs.append(args)


    def on_validation_epoch_end(self, prefix='val'):
        import sed_scores_eval
        import pandas as pd
        import scipy
        from tacos.datasets.audioset_helpers import as_strong_train_classes, as_strong_eval_classes
        from tacos.datasets.audioset import category_to_audioset

        predictions = {
            fn: e for b in self.validation_outputs for fn, e in zip(b['fname'], b['audio_embeddings'])
        }

        durations = {
            fn: d for b in self.validation_outputs for fn, d in zip(b['fname'], b['duration'])
        }

        ground_truth = {
            fn: [(on, off, cap) for on, off, cap in zip(onsets, offsets, captions)] for b in self.validation_outputs
            for fn, onsets, offsets, captions in zip(b['fname'], b['onsets'], b['offsets'], b['captions'])
        }

        prediction_scores = {}

        for fn, prediction in predictions.items():

            similarities = {
                'onset': np.arange(prediction.shape[0]) * 0.04,
                'offset': (np.arange(prediction.shape[0]) + 1) * 0.04
            }
            if self.kwargs["test_on_audioset_full"]:
                for c in set(as_strong_eval_classes).intersection(set(as_strong_train_classes)):
                    index = as_strong_train_classes.index(c)
                    p = (scipy.ndimage.filters.median_filter(prediction[:, index][:, None].cpu().numpy().astype(np.float32), (self.kwargs["median_filter"], 1)).sum(-1))
                    similarities[c] = p
            else:
                for c in category_to_audioset.keys():
                    indices = [as_strong_train_classes.index(ac) for ac in category_to_audioset[c]]
                    p = (scipy.ndimage.filters.median_filter(prediction[:, indices].cpu().numpy().astype(np.float32), (self.kwargs["median_filter"], 1)).sum(-1))
                    similarities[c] = p

            prediction_scores[fn] = pd.DataFrame(similarities)

        from collections import defaultdict

        def clean_events(data):
            cleaned_data = {}

            for filename, events in data.items():
                # Group events by class
                class_to_events = defaultdict(list)
                for onset, offset, cls in events:
                    class_to_events[cls].append((onset, offset, cls))

                merged_events = []

                for cls, cls_events in class_to_events.items():
                    # Sort by onset time
                    cls_events.sort(key=lambda x: x[0])
                    filtered = []

                    for event in cls_events:
                        if not filtered:
                            filtered.append(event)
                        else:
                            last_onset, last_offset, _ = filtered[-1]
                            curr_onset, curr_offset, _ = event

                            # Merge if overlapping or adjacent
                            if curr_onset <= last_offset + 1e-6:
                                new_offset = max(last_offset, curr_offset)
                                filtered[-1] = (last_onset, new_offset, cls)
                            else:
                                filtered.append(event)

                    merged_events.extend(filtered)

                # Sort the final merged events by onset for each file
                cleaned_data[filename] = sorted(merged_events, key=lambda x: x[0])
            return cleaned_data

        import json
        with open("tacos_ground_truth.json", "w") as f:
            json.dump(clean_events({k.split(".")[0]: v for k, v in ground_truth.items()}), f)

        with open("tacos_duration.json", "w") as f:
            json.dump({k.split(".")[0]: v for k, v in durations.items()}, f)

        os.makedirs("predictions", exist_ok=True)
        for fn, df in prediction_scores.items():
            df.to_csv(f"predictions/{fn.split('.')[0]}.csv", index=False)

        psds1 = sed_scores_eval.intersection_based.psds(
            prediction_scores,
            clean_events(ground_truth),
            durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            num_jobs=1
        )
        self.log(f"{prefix}/psds1", psds1[0])
        self.log(f"{prefix}/psds1_macro_averaged", np.array([v for k, v in psds1[1].items()]).mean())
        for c in psds1[1]:
            self.log(f"{prefix}_class/psds1_{c}", psds1[1][c])

        auroc = sed_scores_eval.segment_based.auroc(
            prediction_scores,
            clean_events(ground_truth),
            durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1
        )

        self.log(f"{prefix}/pauroc", auroc[0]["mean"])
        for c in auroc[0]:
            self.log(f"{prefix}_class/pauroc_{c}", auroc[0][c])
        self.validation_outputs.clear()
        return


    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix='test')

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False
        )

        return optimizer

    def lr_scheduler_step(self, batch_idx):

        steps_per_epoch = self.trainer.num_training_batches

        min_lr = self.kwargs['min_lr']
        max_lr = self.kwargs['max_lr']
        current_step = self.current_epoch * steps_per_epoch + batch_idx
        warmup_steps = self.kwargs['warmup_epochs'] * steps_per_epoch
        total_steps = (self.kwargs['warmup_epochs'] + self.kwargs['rampdown_epochs']) * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            # Linear warmup
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            # Cosine decay
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            # Constant learning rate
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)

