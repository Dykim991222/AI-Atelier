
# test_model.py

import torch

from load_data import load_data
from model import load_model


def main():

    # 1. 실행 장치 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("사용 장치:", device)

    # 2. 원본 데이터 불러오기
    images, labels, image_infos = load_data()

    print("이미지 크기:", images.shape)
    print("정답 크기:", labels.shape)

    # 3. ResNet18 모델 불러오기
    model = load_model(device)

    # 4. 예측 수행
    correct = 0
    total = len(images)
    batch_size = 16

    for start in range(0, total, batch_size):

        batch_images = images[start:start + batch_size].to(device)
        batch_labels = labels[start:start + batch_size].to(device)

        with torch.no_grad():
            outputs = model(batch_images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == batch_labels).sum().item()

        # 이미지별 결과 출력
        for i, pred in enumerate(predictions):
            index = start + i
            true_label = batch_labels[i].item()
            pred_label = pred.item()

            status = "정답" if pred_label == true_label else "오답"

            print(
                f"[{index + 1:03d}] "
                f"{status} | "
                f"정답: {true_label} | "
                f"예측: {pred_label}"
            )

    # 5. 최종 정확도
    accuracy = correct / total * 100

    print("\n===== 최종 결과 =====")
    print(f"전체 이미지: {total}장")
    print(f"정답 이미지: {correct}장")
    print(f"오답 이미지: {total - correct}장")
    print(f"정확도: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
