import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from rps.data import RPSDataModule
from rps.module import RPSModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    datamodule = RPSDataModule(
        cfg.data.train_data_dir,
        cfg.data.test_data_dir,
        cfg.data.val_data_dir,
        cfg.data.train_batch_size,
        cfg.data.test_batch_size,
    )

    model = RPSModule(num_classes=cfg.model.num_classes)

    logger = TensorBoardLogger("tb_logs", name=cfg.logging.model_name)
    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # Сохранение модели (как в примере)
    torch.save(model.state_dict(), cfg.output_file)


if __name__ == "__main__":
    train()
