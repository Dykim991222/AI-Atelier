
# fgsm.py

import torch
import torch.nn.functional as F

from load_data import load_data
from model import load_model


def fgsm_attack(model, images, labels, epsilon=8/255):

    # 원본 이미지 복사
    x = images.clone().detach()
    x.requires_grad_(True)

    # 1. 모델 예측
    outputs = model(x)

    # 2. Cross Entropy Loss 계산
    loss = F.cross_entropy(outputs, labels)

    # 3. 입력 이미지에 대한 Gradient 계산
    gradient = torch.autograd.grad(loss, x)[0]

    # 4. Gradient의 부호를 이용해 노이즈 생성
    noise = epsilon * gradient.sign()

    # 5. 공격 이미지 생성
    adv_images = torch.clamp(x + noise, 0, 1)

    return adv_images.detach()


def main():

    # 실행 장치 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("사용 장치:", device)

    # 데이터 불러오기
    images, labels, image_infos = load_data()

    # 모델 불러오기
    model = load_model(device)

    total = len(images)
    success = 0
    batch_size = 16

    for start in range(0, total, batch_size):

        batch_images = images[start:start + batch_size].to(device)
        batch_labels = labels[start:start + batch_size].to(device)

        # FGSM 공격
        adv_images = fgsm_attack(
            model,
            batch_images,
            batch_labels
        )

        # 공격 이미지 재분류
        with torch.no_grad():
            outputs = model(adv_images)
            predictions = outputs.argmax(dim=1)

        # 오분류 이미지 수 계산
        success += (predictions != batch_labels).sum().item()

        for i, pred in enumerate(predictions):
            index = start + i

            status = (
                "공격 성공"
                if pred.item() != batch_labels[i].item()
                else "공격 실패"
            )

            print(
                f"[{index + 1:03d}] "
                f"{status} | "
                f"정답: {batch_labels[i].item()} | "
                f"예측: {pred.item()}"
            )

    attack_rate = success / total * 100

    print("\n===== FGSM 공격 결과 =====")
    print(f"전체 이미지: {total}장")
    print(f"공격 성공: {success}장")
    print(f"공격 실패: {total - success}장")
    print(f"공격 성공률: {attack_rate:.2f}%")


if __name__ == "__main__":
    main()
