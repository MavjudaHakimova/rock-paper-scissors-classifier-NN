import lightning as L
import torch
import torch.nn.functional as F
from rps.model import FeatureExtractor
from torchmetrics import Accuracy, F1Score


class RPSModule(L.LightningModule):
    def __init__(self, num_classes: int = 3, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = FeatureExtractor()
        self.classifier_head = torch.nn.Linear(1280, num_classes)

        # Метрики для всех стадий
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier_head(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Обновляем метрики
        self.train_accuracy.update(preds, labels)
        self.train_f1.update(preds, labels)

        self.log("train_loss", loss, prog_bar=True)

        # ВРЕМЕННЫЕ step-метрики (не epoch)
        self.log("train_acc_step", self.train_accuracy, prog_bar=True, on_step=True)
        self.log("train_f1_step", self.train_f1, prog_bar=True, on_step=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Обновляем метрики
        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Обновляем метрики
        self.test_accuracy.update(preds, labels)
        self.test_f1.update(preds, labels)

        self.log("test_loss", loss, prog_bar=True)

    def on_train_epoch_end(self):
        acc = self.train_accuracy.compute()
        f1 = self.train_f1.compute()
        self.log("train_acc_epoch", acc, prog_bar=True)
        self.log("train_f1_epoch", f1, prog_bar=True)
        self.train_accuracy.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        acc = self.val_accuracy.compute()
        f1 = self.val_f1.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.val_accuracy.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        acc = self.test_accuracy.compute()
        f1 = self.test_f1.compute()
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.test_accuracy.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
