from pathlib import Path
import hydra
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from collections import Counter
from omegaconf import DictConfig

from rps.module import RPSModule

CLASS_NAMES = ["rock", "paper", "scissors"]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def infer(cfg: DictConfig):
    print("Inference на плоской папке...")

    inference_dir = Path("inference_dir")
    if not inference_dir.exists():
        print(f"Создайте {inference_dir}/ и положите туда JPG/PNG файлы")
        return

    device = torch.device("mps" if cfg.trainer.accelerator == "mps" else "cpu")
    model = RPSModule(num_classes=cfg.model.num_classes)
    model.load_state_dict(torch.load(cfg.output_file, weights_only=True))
    model.to(device).eval()

    print(f"Модель: {cfg.output_file}")

    predictions = predict_folder(model, inference_dir, device)

    print(f"Обработано: {len(predictions)} изображений")
    class_counts = Counter([p["class"] for p in predictions])
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    print("\n ПРЕДСКАЗАНИЯ:")
    for pred in predictions[:10]:
        print(f"  {pred['filename']}: {pred['class']} ({pred['confidence']:.1%})")

    print("Готово!")


def predict_folder(model, folder_path: Path, device):
    """Предсказывает классы для всех изображений в папке"""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    predictions = []

    images = (
        list(folder_path.glob("*.jpg"))
        + list(folder_path.glob("*.jpeg"))
        + list(folder_path.glob("*.png"))
    )
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs.max().item()

        predictions.append(
            {
                "filename": img_path.name,
                "class": CLASS_NAMES[pred_idx],
                "confidence": confidence,
                "probs": probs.cpu().numpy()[0],
            }
        )

    return predictions


if __name__ == "__main__":
    infer()
