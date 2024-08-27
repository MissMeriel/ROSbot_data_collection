# NICE ORGANIZED CLEAN: https://github.com/JDScript/COMP3340-gp/tree/74b80c482420420e4f21aa55c512c07477712cd4
import sys, os
# sys.path.append("../models/")
import torch
import torch.nn as nn
from vit import vit_b_16, ViT_B_16_Weights


class ViT_B_16(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Sequential()

    def forward(self, x, return_weights=False):
        out, weights = self.backbone(x)

        if return_weights:
            return out, weights

        return out


class LinearHead(nn.Module):
    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        # self.softmax = nn.LogSoftmax(1)

    def forward(self, x: torch.Tensor):
        out = self.fc(x)
        # out = self.softmax(out)
        x = torch.tanh(x)
        return out


if __name__ == '__main__':
    backbone = ViT_B_16()
    head = LinearHead(in_features=768, out_features=1)
    backbone.backbone.heads = head
    print(backbone)
    out = backbone(torch.rand((1, 3, 224, 224)))
    print(f"{out.shape=}, {out=}")