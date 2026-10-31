
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from load_data import load_data
from model import load_model
from targeted_pgd import targeted_pgd_attack


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PNG_DIR = OUTPUT_DIR / "pgd_png"
ZIP_PATH = OUTPUT_DIR / "c10_pgd_submission.zip"


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    images, labels, targets, image_infos = load_data()
    model = load_model(device)

    PNG_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    success_count = 0
    max_difference = 0

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size].to(device)
        batch_targets = targets[start:start + batch_size].to(device)

        # PGD 공격 수행
        adv_images = targeted_pgd_attack(
            model,
            batch_images,
            batch_targets,
            epsilon=16/255,
            alpha=2/255,
            steps=20
        )

        for j in range(len(adv_images)):
            index = start + j
            filename = image_infos[index]["file"]

            # 원본을 uint8 RGB 픽셀로 변환
            original = (
                images[index].permute(1, 2, 0).numpy() * 255
            ).round().astype(np.uint8)

            # 공격 결과를 uint8 RGB 픽셀로 변환
            adversarial = (
                adv_images[j].cpu().permute(1, 2, 0).numpy() * 255
            ).round().clip(0, 255).astype(np.uint8)

            # 원본 이미지 대비 픽셀 차이 검증
            difference = np.abs(
                adversarial.astype(np.int16)
                - original.astype(np.int16)
            )

            current_max = int(difference.max())
            max_difference = max(max_difference, current_max)

            if current_max > 16:
                raise ValueError(
                    f"{filename}: 픽셀 변경량 {current_max} > 16"
                )

            # PNG 저장
            Image.fromarray(adversarial, mode="RGB").save(
                PNG_DIR / filename
            )

        print(f"저장 완료: {min(start + batch_size, len(images))}/100")

    # 저장한 PNG를 다시 로드하여 실제 제출 이미지 검증
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_files = image_infos[start:start + batch_size]
            batch_targets = targets[start:start + batch_size].to(device)

            saved_images = []

            for info in batch_files:
                with Image.open(PNG_DIR / info["file"]) as img:
                    arr = np.asarray(img.convert("RGB")).copy()

                saved_images.append(
                    torch.from_numpy(arr).permute(2, 0, 1).float() / 255
                )

            batch = torch.stack(saved_images).to(device)
            predictions = model(batch).argmax(dim=1)

            success_count += (
                predictions == batch_targets
            ).sum().item()

    # ZIP 생성
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in image_infos:
            filename = info["file"]
            zf.write(PNG_DIR / filename, arcname=filename)

    print("\n===== 최종 검증 결과 =====")
    print(f"저장 이미지: {len(image_infos)}장")
    print(f"최대 픽셀 변경량: {max_difference}/16")
    print(f"PNG 재로드 후 대리 모델 성공: {success_count}/100")
    print(f"제출 파일: {ZIP_PATH}")


if __name__ == "__main__":
    main()
