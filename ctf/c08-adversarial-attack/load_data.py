
# load_data.py

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor


def load_data():
    # 1. 원본 이미지 폴더 경로 설정
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "c08-originals"

    # 2. manifest.json 읽기
    manifest_path = DATA_DIR / "manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 3. 이미지 정보 가져오기
    image_infos = manifest["images"]

    # 4. 이미지와 정답을 저장할 리스트
    images = []
    labels = []

    # 5. 이미지 100장 읽기
    for info in image_infos:
        filename = info["file"]
        label = info["class_index"]

        image_path = DATA_DIR / filename
        image = Image.open(image_path).convert("RGB")
        image_tensor = ToTensor()(image)

        images.append(image_tensor)
        labels.append(label)

    # 6. Tensor 변환
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    # 7. 결과 반환
    return images, labels, image_infos


# 이 파일을 직접 실행할 때만 동작
if __name__ == "__main__":
    images, labels, image_infos = load_data()

    print("전체 이미지 수:", len(image_infos))
    print("이미지 Tensor 크기:", images.shape)
    print("정답 Tensor 크기:", labels.shape)
    print("이미지 픽셀 최솟값:", images.min().item())
    print("이미지 픽셀 최댓값:", images.max().item())
    print("첫 번째 정답:", labels[0].item())
    print("첫 번째 파일명:", image_infos[0]["file"])
