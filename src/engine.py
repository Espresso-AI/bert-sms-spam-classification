import datetime
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, OrderedDict
from transformers import AdamW
from src.utils.lr_scheduler import *
from sklearn.metrics import classification_report


class SeqCls_Engine(pl.LightningModule):

    def __init__(
            self,
            model,
            ckpt_path: Optional[str] = None,
            freeze_base: bool = False,
            lr: float = None,
            weight_decay: float = 0.01,
            adam_epsilon: float = 1e-8,
            num_warmup_steps: int = None,
            num_training_steps: int = None,
            lr_init_eps: float = 0.1,
            save_result: bool = False,
    ):
        super().__init__()

        self.model = model
        self.ckpt_path = ckpt_path
        self.freeze_base = freeze_base
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_init_eps = lr_init_eps
        self.save_result = save_result

        self.prepare_training()


    def prepare_training(self):
        self.model.train()

        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path)
            assert isinstance(checkpoint, OrderedDict), \
                'please load lightning-format checkpoints'
            assert next(iter(checkpoint)).split('.')[0] != 'model', \
                'this is only for loading the model checkpoints'
            self.model.load_state_dict(checkpoint)

        if self.freeze_base:
            for p in self.model.base_model.parameters():
                p.requires_grad = False


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optim_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optim_params, lr=self.lr, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            self.num_warmup_steps,
            self.num_training_steps,
            self.lr_init_eps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'monitor': 'val_acc',
                'interval': 'epoch'
            }
        }

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(**inputs, labels=labels)

        loss = outputs['loss']
        preds = outputs['prediction']
        accuracy = (preds == labels).sum() / labels.size(0)

        self.log('train_step_loss', loss, prog_bar=True)
        return {'loss': loss, 'accuracy': accuracy}


    def training_epoch_end(self, train_steps):
        total_loss, total_acc = [], []
        for output in train_steps:
            total_loss.append(output['loss'])
            total_acc.append(output['accuracy'])

        train_loss = torch.tensor(total_loss).mean()
        train_acc = torch.tensor(total_acc).mean()

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_acc', train_acc, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(**inputs, labels=labels)

        loss = outputs['loss']
        preds = outputs['prediction']
        accuracy = (preds == labels).sum() / labels.size(0)

        return loss, accuracy


    def validation_epoch_end(self, val_steps):
        total_loss, total_acc = [], []
        for loss, acc in val_steps:
            total_loss.append(loss)
            total_acc.append(acc)

        val_loss = torch.tensor(total_loss).mean()
        val_acc = torch.tensor(total_acc).mean()

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(**inputs)

        return outputs['prediction'], labels


    def test_epoch_end(self, test_steps):
        total_preds, total_labels = [], []

        for labels, preds in test_steps:
            total_preds.append(preds)
            total_labels.append(labels)

        total_preds = torch.cat(total_preds, dim=0).detach().cpu().numpy()
        total_labels = torch.cat(total_labels, dim=0).detach().cpu().numpy()

        if self.save_result:
            result_pd = pd.DataFrame(
                {'label': total_labels,
                 'prediction': total_preds}
            )
            time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
            result_pd.to_csv('./test_result-' + time + '.csv', index=False)

        result = classification_report(
            total_labels,
            total_preds,
            target_names=['ham', 'spam'],
            digits=4
        )
        print(result)
