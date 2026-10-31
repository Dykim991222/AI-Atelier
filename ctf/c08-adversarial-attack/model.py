
# model.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

        # ImageNet 정규화 값
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.register_buffer(
            "mean",
            torch.tensor(mean).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(std).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return (x - self.mean) / self.std


def load_model(device):
    # 사전 학습된 ResNet18
    resnet = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )

    # 정규화 + ResNet18
    model = nn.Sequential(
        Normalize(),
        resnet
    )

    model = model.to(device)
    model.eval()

    return model
