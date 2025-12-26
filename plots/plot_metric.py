import matplotlib.pyplot as plt
import os


def save_metrics_plots(trainer, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = trainer.logged_metrics

    plt.figure(figsize=(10, 4))
    plt.plot(metrics.get("train_loss", []), label="Train Loss")
    plt.plot(metrics.get("val_loss", []), label="Val Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(f"{output_dir}/01_loss_curves.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(metrics.get("train_acc", []), label="Train Acc")
    plt.plot(metrics.get("val_acc", []), label="Val Acc")
    plt.plot(metrics.get("train_f1", []), label="Train F1")
    plt.plot(metrics.get("val_f1", []), label="Val F1")
    plt.title("Accuracy & F1 Curves")
    plt.legend()
    plt.savefig(f"{output_dir}/02_accuracy_f1_curves.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(metrics.get("learning_rate", []))
    plt.title("Learning Rate Schedule")
    plt.xlabel("Steps")
    plt.ylabel("LR")
    plt.savefig(f"{output_dir}/03_learning_rate.png")
    plt.close()
