
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return (x - self.mean) / self.std


def load_model(device):
    weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(weights=weights)

    model = nn.Sequential(
        Normalize(),
        model
    )

    model = model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = load_model(device)

    print("사용 장치:", device)
    print("대리 모델: ResNet18")
    print("모델 로드 완료")
