
from pathlib import Path
import zipfile

import numpy as np
import torch
from PIL import Image

from load_data import load_data
from model import load_model
from fgsm import fgsm_attack


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PNG_DIR = OUTPUT_DIR / "fgsm_png"
ZIP_PATH = OUTPUT_DIR / "c09_fgsm_submission.zip"

EPSILON = 8
BATCH_SIZE = 16


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    images, labels, image_infos = load_data()
    model = load_model(device)

    PNG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. FGSM 공격 이미지 생성 및 PNG 저장
    for start in range(0, len(images), BATCH_SIZE):
        batch_images = images[start:start + BATCH_SIZE].to(device)
        batch_labels = labels[start:start + BATCH_SIZE].to(device)

        adv_images = fgsm_attack(
            model,
            batch_images,
            batch_labels,
            epsilon=EPSILON / 255
        )

        for j in range(len(adv_images)):
            index = start + j
            filename = image_infos[index]["file"]

            adv_array = (
                adv_images[j]
                .cpu()
                .permute(1, 2, 0)
                .numpy() * 255
            ).round().clip(0, 255).astype(np.uint8)

            Image.fromarray(adv_array).save(PNG_DIR / filename)

        print(f"PNG 저장: {min(start + BATCH_SIZE, len(images))}/{len(images)}")

    # 2. 저장된 PNG의 픽셀 변경량 검증
    max_diff = 0
    success_count = 0

    for start in range(0, len(images), BATCH_SIZE):
        saved_tensors = []
        batch_infos = image_infos[start:start + BATCH_SIZE]

        for j, info in enumerate(batch_infos):
            index = start + j
            filename = info["file"]

            original_array = (
                images[index]
                .permute(1, 2, 0)
                .numpy() * 255
            ).round().astype(np.uint8)

            with Image.open(PNG_DIR / filename) as img:
                saved_array = np.array(img.convert("RGB"))

            diff = np.abs(
                saved_array.astype(np.int16)
                - original_array.astype(np.int16)
            )

            current_max = int(diff.max())
            max_diff = max(max_diff, current_max)

            if current_max > EPSILON:
                raise ValueError(
                    f"{filename}: 최대 픽셀 변경량 {current_max} > {EPSILON}"
                )

            saved_tensors.append(
                torch.from_numpy(saved_array.copy())
                .permute(2, 0, 1)
                .float() / 255
            )

        # 3. PNG 재로드 후 대리 모델에서 공격 성공 확인
        batch = torch.stack(saved_tensors).to(device)
        batch_labels = labels[start:start + BATCH_SIZE].to(device)

        with torch.no_grad():
            predictions = model(batch).argmax(dim=1)

        success_count += (predictions != batch_labels).sum().item()

    # 4. ZIP 압축
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in image_infos:
            filename = info["file"]
            zf.write(PNG_DIR / filename, arcname=filename)

    print("\n===== 최종 검증 =====")
    print(f"PNG 이미지: {len(image_infos)}장")
    print(f"최대 픽셀 변경량: {max_diff}/{EPSILON}")
    print(f"PNG 저장 후 대리 모델 오분류: {success_count}/{len(images)}")
    print(f"제출 ZIP: {ZIP_PATH}")


if __name__ == "__main__":
    main()
