import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from load_data import load_data
from model import load_model


def fgsm_attack(model, images, labels, epsilon=8/255):
    x = images.clone().detach()
    x.requires_grad_(True)

    outputs = model(x)
    loss = F.cross_entropy(outputs, labels)

    gradient = torch.autograd.grad(loss, x)[0]
    adv_images = x + epsilon * gradient.sign()
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


def save_tensor_as_png(tensor_image, save_path):
    # tensor_image: [3,224,224], range [0,1]
    arr = (tensor_image.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    img = Image.fromarray(arr)
    img.save(save_path)


def load_png_as_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.tensor(arr, dtype=torch.float32)


def max_pixel_diff(original_tensor, saved_tensor):
    # 둘 다 [3,224,224], range [0,1]
    orig = (original_tensor.cpu().numpy() * 255.0).round().astype(np.int16)
    adv = (saved_tensor.cpu().numpy() * 255.0).round().astype(np.int16)
    diff = np.abs(orig - adv)
    return diff.max()


def main():
    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("사용 장치:", device)

    # 경로
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs" / "fgsm_png"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터/모델 로드
    images, labels, image_infos = load_data()
    model = load_model(device)

    total = len(images)
    batch_size = 16

    # 1. FGSM 생성 + PNG 저장
    for start in range(0, total, batch_size):
        batch_images = images[start:start+batch_size].to(device)
        batch_labels = labels[start:start+batch_size].to(device)

        adv_images = fgsm_attack(model, batch_images, batch_labels)

        for i in range(len(adv_images)):
            idx = start + i
            filename = image_infos[idx]["file"]
            save_path = output_dir / filename
            save_tensor_as_png(adv_images[i], save_path)

    print("PNG 저장 완료")

    # 2. 저장된 PNG 다시 읽어서 검증
    success = 0
    max_diff_overall = 0
    violating_files = []

    reloaded_images = []
    reloaded_labels = []

    for idx, info in enumerate(image_infos):
        filename = info["file"]
        label = info["class_index"]

        saved_path = output_dir / filename
        saved_tensor = load_png_as_tensor(saved_path)
        original_tensor = images[idx]

        diff = max_pixel_diff(original_tensor, saved_tensor)

        if diff > max_diff_overall:
            max_diff_overall = diff

        if diff > 8:
            violating_files.append((filename, diff))

        reloaded_images.append(saved_tensor)
        reloaded_labels.append(label)

    reloaded_images = torch.stack(reloaded_images)
    reloaded_labels = torch.tensor(reloaded_labels, dtype=torch.long)

    # 3. 저장된 PNG 기준 성공률 측정
    for start in range(0, total, batch_size):
        batch_images = reloaded_images[start:start+batch_size].to(device)
        batch_labels = reloaded_labels[start:start+batch_size].to(device)

        with torch.no_grad():
            outputs = model(batch_images)
            preds = outputs.argmax(dim=1)

        success += (preds != batch_labels).sum().item()

    attack_rate = success / total * 100

    print("\n===== 저장 후 검증 결과 =====")
    print(f"전체 이미지: {total}")
    print(f"공격 성공 수: {success}")
    print(f"공격 성공률: {attack_rate:.2f}%")
    print(f"최대 픽셀 차이: {max_diff_overall}")

    if violating_files:
        print("\n[경고] 픽셀 차이 8 초과 파일:")
        for fname, diff in violating_files:
            print(f"{fname} | diff={diff}")
    else:
        print("\n모든 파일이 L∞ <= 8 조건을 만족합니다.")


if __name__ == "__main__":
    main()