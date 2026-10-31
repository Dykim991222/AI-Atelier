
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor


DATA_DIR = Path(__file__).resolve().parent / "data" / "c10-originals"


def load_data():
    manifest_path = DATA_DIR / "manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images = []
    labels = []
    targets = []
    image_infos = manifest["images"]

    to_tensor = ToTensor()

    for info in image_infos:
        image_path = DATA_DIR / info["file"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            images.append(to_tensor(image))

        labels.append(info["class_index"])
        targets.append(info["target_index"])

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return images, labels, targets, image_infos


if __name__ == "__main__":
    images, labels, targets, image_infos = load_data()

    print("이미지 Shape:", images.shape)
    print("정답 레이블 Shape:", labels.shape)
    print("표적 레이블 Shape:", targets.shape)
    print("첫 번째 파일:", image_infos[0]["file"])
    print("첫 번째 정답:", labels[0].item())
    print("첫 번째 표적:", targets[0].item())
