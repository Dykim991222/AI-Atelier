
import torch

from load_data import load_data
from model import load_model


def main():
    # 1. 실행 장치 선택
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # 2. 이미지와 정답 레이블 불러오기
    images, labels, image_infos = load_data()

    # 3. ResNet18 대리 모델 불러오기
    model = load_model(device)

    # 4. 이미지 100장 예측
    predictions = []
    batch_size = 16

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size].to(device)

            logits = model(batch_images)
            predicted = logits.argmax(dim=1)

            predictions.append(predicted.cpu())

    predictions = torch.cat(predictions)

    # 5. 정답 레이블과 예측 결과 비교
    correct = predictions == labels
    correct_count = correct.sum().item()

    accuracy = correct_count / len(labels) * 100

    print(f"사용 장치: {device}")
    print(f"전체 이미지: {len(images)}장")
    print(f"정답 분류: {correct_count}장")
    print(f"원본 분류 정확도: {accuracy:.2f}%")

    # 6. 처음 10장의 예측 결과 출력
    print("\n처음 10장 예측 결과")
    print("-" * 60)

    for i in range(min(10, len(images))):
        print(
            f"[{i+1:02d}] "
            f"정답: {labels[i].item():3d} | "
            f"예측: {predictions[i].item():3d} | "
            f"정답 여부: {bool(correct[i].item())}"
        )


if __name__ == "__main__":
    main()
