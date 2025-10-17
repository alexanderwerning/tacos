import copy
import math
import string
from typing import Any
import math

import numpy as np
import torch
import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaModel
from triton.language.semantic import reduction

from tacos.models.sed import ASITSEDWrapper

import torch
import torch.nn.functional as F

class AudioRetrievalModel(pl.LightningModule):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__()
        self.save_hyperparameters(kwargs)

        # audio encoder
        self.audio_embedding_model = ASITSEDWrapper(checkpoint=kwargs['audio_embedding_checkpoint'])
        self.audio_projection = torch.nn.Linear(768, 1024)

        # text encoder
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.text_embedding_model = RobertaModel.from_pretrained(
            'roberta-base' if kwargs['roberta_base'] else 'roberta-large',
            add_pooling_layer=False,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            output_hidden_states=False
        )
        self.text_projection = torch.nn.Linear(768 if kwargs['roberta_base'] else 1024, 1024)

        # temperature parameter
        initial_tau = torch.zeros((1,)) + kwargs['initial_tau']
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=kwargs['tau_trainable'])

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
                self.text_embedding_model = torch.compile(self.text_embedding_model)
                self.audio_embedding_model.model.model = torch.compile(self.audio_embedding_model.model.model)

    def forward(self, batch) -> Any:

        # embed audio & text
        text_embeddings = self.forward_text(batch)
        audio_embeddings = self.forward_audio(batch)

        return audio_embeddings, text_embeddings


    def forward_audio(self, batch):

        audio_embedding_sequence = self.audio_embedding_model(
            batch['audio'].mean(1),
            duration=torch.tensor(batch["duration"])/ 30
        ).permute(0,2,1) # forward

        batch['audio_embedding'] = audio_embedding_sequence

        # mask embeddings from padded empty audio parts
        audio_embeddings = []
        for i, duration in enumerate(batch['duration']):
            step_size = 29.82 / audio_embedding_sequence.shape[1]
            offset = math.ceil(duration / step_size)
            audio_embedding = audio_embedding_sequence[i, :offset].mean(0)

            audio_embedding = self.audio_projection(audio_embedding)  # project to same dimension
            audio_embedding = torch.nn.functional.normalize(audio_embedding, p=2, dim=-1)  # normalize
            audio_embeddings.append(audio_embedding)

        return audio_embeddings

    def forward_text(self, batch):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        batch['captions'] = [
            [c.lower().translate(str.maketrans('', '', string.punctuation)).strip() for c in all_c] for all_c in batch['captions']
        ]

        unique_captions = list({c for all_c in batch['captions'] for c in all_c})

        tokenized = self.tokenizer(
            unique_captions,
            add_special_tokens=True,
            padding='max_length',
            return_tensors='pt',
            max_length=32,
            truncation=True
        )

        token_embeddings = self.text_embedding_model(
            input_ids=tokenized['input_ids'].to(device),
            attention_mask=tokenized['attention_mask'].to(device)
        )[0]
        # select first token of sequence
        sentence_features = token_embeddings[:, 0, :]
        # project
        sentence_features = self.text_projection(sentence_features)
        # normalize
        sentence_features = torch.nn.functional.normalize(sentence_features, p=2, dim=-1)

        sentence_features = {k:v for k,v in zip(unique_captions, sentence_features)}

        sentence_features = [
            torch.stack([sentence_features[c] for c in all_c]) for all_c in batch['captions']
        ]

        return sentence_features

    def contrastive_region_loss(self, batch):
        audio_embedding = self.audio_projection(batch["audio_embedding"])
        audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)  # (B, T, D)
        step_size = 29.82 / audio_embedding.shape[1]

        with torch.set_grad_enabled(not self.kwargs["freeze_text_encoder"]):
            te = self.forward_text({'captions': batch['captions_strong']})  # (B, N, D)

        loss = 0
        temperature = self.kwargs.get('strong_tau', 0.07)
        total_regions = 0

        unique_indices = [list({s: j for j, s in enumerate(b)}.values()) for b in batch['captions_strong']]

        for i in range(len(te)):

            text_other = [te[idx][unique_indices[idx]] for idx in np.arange(len(te)) if idx != i]
            text_other = torch.concat(text_other, dim=0)

            for j, (on, off) in enumerate(zip(batch['onsets'][i], batch['offsets'][i])):
                on_idx = math.floor(on / step_size)
                off_idx = math.ceil(off / step_size)

                text = torch.cat([te[i][j:j+1], text_other], dim=0)

                C_ta = (text[:, None, :] * audio_embedding[i, on_idx:off_idx][None, :, :]).sum(-1)/ temperature
                C_ta = torch.log_softmax(C_ta, dim=0)
                loss += -C_ta[0, :].mean()
                total_regions += 1

        return loss / total_regions

    def training_step(self, batch, batch_idx):

        self.lr_scheduler_step(batch_idx)

        audio_embeddings, text_embeddings = self.forward(batch) # batch 1: sound IDs ['204046', '266329']; ['Paper_Parchment_Rustling.wav', 'metalTunnel.wav']

        a = torch.stack(audio_embeddings)
        t = torch.concatenate(text_embeddings)

        C = torch.matmul(a, t.T)
        C = C / torch.abs(self.tau)

        sample_ids = np.array([ i for i, cs in enumerate(batch["captions"]) for c in cs])
        batch_mask = sample_ids[:, None] == sample_ids[None, :]
        # batch_mask = T_m & batch_mask
        np.fill_diagonal(batch_mask, False)
        C[batch_mask] = -torch.inf

        # compute P(a|t) and P(t|a)
        C_audio = torch.log_softmax(C, dim=0)
        C_text = torch.log_softmax(C, dim=1)

        # prediction target
        I = torch.eye(C_audio.shape[0])

        loss = -0.5 * (C_audio[torch.where(I)].mean() + C_text[torch.where(I)].mean())

        region_loss = self.contrastive_region_loss(batch) if 'captions_strong' in batch else 0
        self.log("train/region_loss", region_loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        self.log("train/loss", loss, batch_size=len(audio_embeddings), sync_dist=True, prog_bar=True)
        self.log('train/tau', torch.abs(self.tau), sync_dist=True)

        return loss * self.kwargs["weak_weight"] + region_loss  * self.kwargs["strong_weight"]

    def validation_step(self, batch, batch_idx):


        if batch['dataset'][0] == 'clotho':
            audio_embeddings, text_embeddings = self.forward(batch)
            audio_embeddings = torch.stack(audio_embeddings)
            text_embeddings = torch.concatenate(text_embeddings)

            args = {
                'audio_embeddings': copy.deepcopy(audio_embeddings.detach()),
                'text_embeddings': copy.deepcopy(text_embeddings.detach()),
                'caption': [c[0] for c in batch['captions']],
                'path': batch['fname'],
                'dataset': batch['dataset']
            }

            self.validation_outputs.append(args)
        elif batch['dataset'][0] == 'audioset_strong':
            batch['captions'] = batch['captions_strong']
            audio_embeddings, text_embeddings = self.forward(batch)
            audio_embeddings = self.audio_projection(batch['audio_embedding'])
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings, p=2, dim=-1)

            if 'captions_strong' in batch:
                self.log("val/region_loss", self.contrastive_region_loss(batch))

            args = {
                'audio_embeddings': copy.deepcopy(audio_embeddings.detach().cpu()),
                'text_embeddings': [copy.deepcopy(te.detach().cpu()) for te in text_embeddings],
                'captions': [c for c in batch['captions']],
                'path': batch['fname'],
                'onsets': batch['onsets'],
                'offsets': batch['offsets'],
                'duration': batch['duration'],
                'dataset': batch['dataset'],
                'fname': batch['fname']
            }

            self.validation_outputs.append(args)

    def audioset_strong_eval(self, prefix="val"):
        import sed_scores_eval
        import pandas as pd

        class_embeddings = {
            ci: ei for b in self.validation_outputs for c, e in zip(b['captions'], b['text_embeddings']) for ci, ei in zip(c, e)
        }

        predictions = {
           fn: e for b in self.validation_outputs for fn, e in zip(b['fname'], b['audio_embeddings'])
        }

        durations = {
            fn: d for b in self.validation_outputs for fn, d in zip(b['fname'], b['duration'])
        }

        ground_truth = {
            fn: [(on, off, cap) for on, off, cap in zip(onsets, offsets, captions)] for b in self.validation_outputs for fn, onsets, offsets, captions in zip(b['fname'], b['onsets'], b['offsets'], b['captions'])
        }

        prediction_scores = {}

        for fn, prediction in predictions.items():

            similarities = {
                'onset': np.arange(prediction.shape[0]) * 0.02,
                'offset': (np.arange(prediction.shape[0]) + 1) * 0.02
            }

            for n, c in class_embeddings.items():
                similarities[n] = (predictions[fn] @ c[: None]).cpu().numpy()

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

    def on_validation_epoch_end(self, prefix='val'):
        outputs = self.validation_outputs

        if outputs[-1]['dataset'][0] == 'audioset_strong':
            self.audioset_strong_eval(prefix=prefix)
            self.validation_outputs.clear()
            return

        # concatenate metadata
        paths = np.array([p for b in outputs for p in b['path']])
        captions = np.array([p for b in outputs for p in b['caption']])

        # audios in clotho can have five captions
        # this snippet discards every occurrence of a duplicate audio
        #
        target = [] # prediction targets for later
        select = [] # indices of the first occurrence for later
        first_occurrence = {} # temporary cache to keep track of first occurrences
        for i, p in enumerate(paths): # iterate over all paths
            index = first_occurrence.get(p)
            if index is None:  # First time seeing this path
                index = len(first_occurrence)
                first_occurrence[p] = index
                select.append(i) # these audios will be selected
            target.append(index) # all paths need a target - choose the correct one
        paths = paths[select]

        # concatenate embeddings
        audio_embeddings = torch.cat([o['audio_embeddings'] for o in outputs])[select]# only select unique audios
        text_embeddings = torch.cat([o['text_embeddings'] for o in outputs])

        # concatenate global ranking
        C = torch.matmul(text_embeddings, audio_embeddings.T)

        # get top 10
        top_ten = C.topk(10, dim=1)[1].detach().cpu().numpy()
        target = np.array(target)

        # recall metrics
        r_1 = (top_ten[:, :1] == target[:, None]).sum(axis=1).mean()
        r_5 = (top_ten[:, :5] == target[:, None]).sum(axis=1).mean()
        r_10 = (top_ten == target[:, None]).sum(axis=1).mean()

        # mAP@10
        AP = 1 / ((top_ten == target[:, None]).argmax(axis=1) + 1)
        AP[~(top_ten == target[:, None]).any(axis=1)] = 0
        mAP = AP.mean()

        # log retrieval performance
        self.log(f'{prefix}/R@1', r_1)
        self.log(f'{prefix}/R@5', r_5)
        self.log(f'{prefix}/R@10', r_10)
        self.log(f'{prefix}/mAP@10', mAP)

        # empty cached batches from validation loop
        self.validation_outputs.clear()


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

