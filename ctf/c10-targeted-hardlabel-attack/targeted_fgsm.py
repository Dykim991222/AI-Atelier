
import torch
import torch.nn.functional as F

from load_data import load_data
from model import load_model


def targeted_fgsm_attack(model, images, targets, epsilon=16/255):
    x = images.detach().clone().requires_grad_(True)

    # 대리 모델의 예측
    logits = model(x)

    # 목표 클래스에 대한 Loss
    loss = F.cross_entropy(logits, targets)

    # 입력 이미지에 대한 Gradient
    gradient = torch.autograd.grad(loss, x)[0]

    # 표적 Loss를 줄이는 방향으로 이동
    x_adv = x - epsilon * gradient.sign()

    # 이미지 픽셀 범위 [0, 1] 유지
    x_adv = x_adv.clamp(0, 1)

    return x_adv.detach()


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    images, labels, targets, image_infos = load_data()
    model = load_model(device)

    batch_size = 16
    success_count = 0

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size].to(device)
        batch_targets = targets[start:start + batch_size].to(device)

        # 공격 이미지 생성
        adv_images = targeted_fgsm_attack(
            model,
            batch_images,
            batch_targets
        )

        # 공격 후 대리 모델의 예측 확인
        with torch.no_grad():
            predictions = model(adv_images).argmax(dim=1)

        success_count += (
            predictions == batch_targets
        ).sum().item()

    success_rate = success_count / len(images) * 100

    print(f"사용 장치: {device}")
    print(f"전체 이미지: {len(images)}장")
    print(f"표적 공격 성공: {success_count}장")
    print(f"대리 모델 표적 공격 성공률: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
