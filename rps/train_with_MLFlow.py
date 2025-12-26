import git
import hydra
import lightning as L
import mlflow
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from plots.plot_metric import save_metrics_plots
from rps.data import RPSDataModule
from rps.module import RPSModule
from rps.utils.dvc_data import download_data


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    download_data()
    print(OmegaConf.to_yaml(cfg))

    try:
        repo = git.Repo(search_parent_directories=True)
        print(f"Git commit: {repo.head.object.hexsha}")
    except git.exc.InvalidGitRepositoryError:
        print("Git не найден")

    datamodule = RPSDataModule(
        cfg.data.train_data_dir,
        cfg.data.test_data_dir,
        cfg.data.val_data_dir,
        cfg.data.train_batch_size,
        cfg.data.test_batch_size,
    )

    model = RPSModule(num_classes=cfg.model.num_classes)

    logger = MLFlowLogger(
        experiment_name="rps-classifier",
        tracking_uri="http://127.0.0.1:8080",
    )

    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=16,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    torch.save(model.state_dict(), cfg.output_file)
    save_metrics_plots(trainer)

    mlflow.log_artifacts("plots/")

    mlflow.log_params(OmegaConf.to_container(cfg))
    mlflow.log_param("git_commit", repo.head.object.hexsha)

    print("Графики сохранены в plots/ и загружены в MLflow!")

    print(f"Модель сохранена: {cfg.output_file}")
    print("MLflow: http://127.0.0.1:8080")


if __name__ == "__main__":
    train()
