
import torch

from load_data import load_data
from model import load_model


def main():
    # 1. 실행 장치 선택
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # 2. 데이터 불러오기
    images, labels, targets, image_infos = load_data()

    # 3. 대리 모델 불러오기
    model = load_model(device)

    # 4. 이미지 분류
    predictions = []

    batch_size = 16

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].to(device)

            logits = model(batch)
            predicted = logits.argmax(dim=1)

            predictions.append(predicted.cpu())

    predictions = torch.cat(predictions)

    # 5. 원본 정답과 비교
    correct = (predictions == labels)
    accuracy = correct.float().mean().item() * 100

    # 6. 표적 클래스와 비교
    target_matches = (predictions == targets)
    target_match_count = target_matches.sum().item()

    print(f"사용 장치: {device}")
    print(f"전체 이미지: {len(images)}장")
    print(f"원본 분류 정확도: {accuracy:.2f}%")
    print(f"원본에서 표적 클래스 예측: {target_match_count}장")

    print("\n처음 10장 예측 결과")
    print("-" * 65)

    for i in range(min(10, len(images))):
        print(
            f"[{i+1:02d}] "
            f"정답: {labels[i].item():3d} | "
            f"예측: {predictions[i].item():3d} | "
            f"표적: {targets[i].item():3d} | "
            f"정답 여부: {bool(correct[i].item())}"
        )


if __name__ == "__main__":
    main()
