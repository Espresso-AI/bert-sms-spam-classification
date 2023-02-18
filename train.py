import argparse
import hydra
import torch.nn as nn
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def train(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    # prepare tokenizer, model and loss function
    loss_fn = Focal_Loss_Classification(**cfg.loss)

    model = SeqCls_Model(
        cfg.base_checkpoint,
        num_classes=2,
        loss_fn=loss_fn,
        classifier_dropout=cfg.classifier_dropout
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_checkpoint)

    # load train and validation datasets
    train_df, val_df = spam_dataframe(cfg.dataset.path, True, **cfg.dataset.df)
    train_dataset = Spam_Dataset(train_df)
    val_dataset = Spam_Dataset(val_df)
    val_df.to_csv('./dataset/spam_val.csv', index=False)
    # val_dataset doesn't involve in optimization
    # it should always be overwritten

    collator = Spam_Collator(tokenizer, cfg.max_seq_len)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=True, collate_fn=collator)

    # config training
    engine = SeqCls_Engine(model, **cfg.engine)
    logger = Another_WandbLogger(**cfg.log, save_artifact=False)
    cfg_trainer = Config_Trainer(cfg.trainer)()

    # run training
    trainer = pl.Trainer(
        **cfg_trainer,
        logger=logger,
        num_sanity_val_steps=0
    )
    logger.watch(engine)

    if 'ckpt_path' in cfg:
        trainer.fit(engine, train_loader, val_loader, ckpt_path=cfg.ckpt_path)
    else:
        trainer.fit(engine, train_loader, val_loader)

    wandb.finish()



if __name__ == "__main__":
    train()

