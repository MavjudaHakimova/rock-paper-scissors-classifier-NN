import shutil
import zipfile
from pathlib import Path
import gdown


def download_data(data_dir: str = "data", force: bool = False) -> None:
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    existing_files = list(data_path.glob("*"))
    print(f"Информация: {len(existing_files)} в {data_path}:")
    for f in existing_files[:5]:
        print(f"  - {f.name}")

    if not force and len(existing_files) > 0:
        print(f"Данные уже есть {data_path}")
        return

    if force or len(existing_files) > 0:
        shutil.rmtree(data_path)
        data_path.mkdir(exist_ok=True)
        print("Директория данных очищена")

    file_id = "1L-dwnigLiWw5_4nNn4yfzHKbZoAX-OdG"
    file_url = f"https://drive.google.com/uc?id={file_id}"

    print("Скачивание RPS архива.")
    archive_path = data_path / "rps_dataset.zip"

    try:
        gdown.download(file_url, str(archive_path), quiet=False)
        print("Скачано!")
    except Exception as e:
        print(f"Скачать не получилось(: {e}")
        return

    print("Разархивирование")
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(data_path)
        archive_path.unlink()
        print("Завершено!")
    except Exception as e:
        print(f"Что-то сломалось(: {e}")
        return

    print("Структуирование данных")

    nested_folders = ["Rock-Paper-Scissors", "RPS", "rps", "dataset"]

    for nested in nested_folders:
        nested_path = data_path / nested
        if nested_path.exists():
            print(f"Found nested folder: {nested}")

            for split in ["train", "test", "validation", "val"]:
                src = nested_path / split
                dst = data_path / split
                if src.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                    shutil.move(str(src), str(dst))
                    print(f"  ✓ Moved {split}/ to data/{split}/")

            shutil.rmtree(nested_path, ignore_errors=True)
            break

    # ФИКС validation - проверяем наличие папок классов
    validation_path = data_path / "validation"
    if validation_path.exists():
        classes = [d.name for d in validation_path.iterdir() if d.is_dir()]
        if len(classes) == 0:
            print("Заполняем validation из train...")

            for cls in ["rock", "paper", "scissors"]:
                (validation_path / cls).mkdir(exist_ok=True)

                train_cls_path = data_path / "train" / cls
                if train_cls_path.exists():
                    images = (
                        list(train_cls_path.glob("*.jpg"))
                        + list(train_cls_path.glob("*.jpeg"))
                        + list(train_cls_path.glob("*.png"))
                    )
                    for img in images[:50]:
                        dest = validation_path / cls / img.name
                        if not dest.exists():
                            shutil.move(str(img), str(dest))
                            print(f"  ✓ {img.name} → validation/{cls}/")

            print("Validation заполнен!")

    print("Структура:")
    for split in ["train", "test", "validation", "val"]:
        split_path = data_path / split
        if split_path.exists():
            classes = [d.name for d in split_path.iterdir() if d.is_dir()]
            class_counts = {c: len(list((split_path / c).glob("*.*"))) for c in classes}
            print(f"  {split}/: {len(classes)} классов - {class_counts}")
        else:
            print(f"  {split}/")

    print("Датасет готов к обучению!")


if __name__ == "__main__":
    download_data(force=True)
