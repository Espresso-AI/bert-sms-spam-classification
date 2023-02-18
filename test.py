import argparse
import hydra
from src import *

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", dest='config_name', default=None, type=str)
args = parser.parse_args()


@hydra.main(version_base=None, config_path='./config/', config_name=args.config_name)
def test(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    model = SeqCls_Model(
        cfg.base_checkpoint,
        num_classes=2,
        loss_fn=None,
        classifier_dropout=cfg.classifier_dropout
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_checkpoint)

    # it should be equal to val_df during training
    test_df = pd.read_csv('./dataset/spam_val.csv')
    test_dataset = Spam_Dataset(test_df)
    collator = Spam_Collator(tokenizer, cfg.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    engine = SeqCls_Engine(model, **cfg.engine)
    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(**cfg_trainer, logger=False)

    if 'ckpt_path' in cfg:
        trainer.test(engine, test_loader, ckpt_path=cfg.ckpt_path)
    else:
        raise RuntimeError('no checkpoint is given')



if __name__ == "__main__":
    test()

