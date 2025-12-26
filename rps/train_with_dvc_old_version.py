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
from pathlib import Path

DVC_AVAILABLE = True


def ensure_data(cfg: DictConfig) -> None:
    data_dir = cfg.data.train_data_dir.rsplit("/", 1)[0]

    if DVC_AVAILABLE:
        try:
            print("Проверяем DVC...")
            from dvc.repo import Repo

            repo = Repo(".")
            repo.pull()
            print("✓ Данные из DVC загружены!")
            return
        except Exception as e:
            print(f"DVC не сработал: {e}. Проверяем наличие data/...")

    # Проверяем, есть ли уже data/ с правильной структурой
    data_path = Path(data_dir)
    if data_path.exists():
        train_path = data_path / "train"
        classes = ["rock", "paper", "scissors"]
        if all((train_path / cls).exists() for cls in classes):
            print("✓ Данные уже готовы!")
            return

    print("Скачиваем через gdown...")
    download_data(force=True)  # Только если НЕТ данных


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    ensure_data(cfg)
    print(OmegaConf.to_yaml(cfg))

    try:
        repo = git.Repo(search_parent_directories=True)
        print(f"Git коммит: {repo.head.object.hexsha}")
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
        experiment_name=cfg.logging.mlflow.experiment_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
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
