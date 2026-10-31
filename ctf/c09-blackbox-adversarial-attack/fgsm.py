
import torch
import torch.nn.functional as F

from load_data import load_data
from model import load_model


def fgsm_attack(model, images, labels, epsilon=8/255):
    # 이미지에 대한 Gradient 계산 활성화
    x = images.detach().clone().requires_grad_(True)

    # 모델 예측 및 정답 기준 Loss 계산
    logits = model(x)
    loss = F.cross_entropy(logits, labels)

    # 입력 이미지에 대한 Gradient
    gradient = torch.autograd.grad(loss, x)[0]

    # 정답 Loss를 증가시키는 방향으로 이동
    x_adv = x + epsilon * gradient.sign()

    # 이미지 픽셀 범위 유지
    x_adv = x_adv.clamp(0, 1)

    return x_adv.detach()


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    images, labels, image_infos = load_data()
    model = load_model(device)

    batch_size = 16
    success_count = 0

    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size].to(device)
        batch_labels = labels[start:start + batch_size].to(device)

        # FGSM 공격 이미지 생성
        adv_images = fgsm_attack(
            model,
            batch_images,
            batch_labels
        )

        # 공격 후 예측
        with torch.no_grad():
            predictions = model(adv_images).argmax(dim=1)

        success_count += (
            predictions != batch_labels
        ).sum().item()

    success_rate = success_count / len(images) * 100

    print(f"사용 장치: {device}")
    print(f"전체 이미지: {len(images)}장")
    print(f"오분류 이미지: {success_count}장")
    print(f"대리 모델 오분류율: {success_rate:.2f}%")


if __name__ == "__main__":
    main()
