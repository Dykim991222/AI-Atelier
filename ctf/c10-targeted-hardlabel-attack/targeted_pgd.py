
import torch
import torch.nn.functional as F

from load_data import load_data
from model import load_model


def targeted_pgd_attack(
    model,
    images,
    targets,
    epsilon=16/255,
    alpha=2/255,
    steps=20
):
    original = images.detach()
    x_adv = original.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, targets)

        gradient = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            # 목표 클래스 Loss 감소
            x_adv = x_adv - alpha * gradient.sign()

            # 원본 이미지에서 ±epsilon 이내로 제한
            x_adv = torch.max(
                torch.min(x_adv, original + epsilon),
                original - epsilon
            )

            # 유효한 이미지 범위 유지
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

        adv_images = targeted_pgd_attack(
            model,
            batch_images,
            batch_targets
        )

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
