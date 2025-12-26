import timm
import torch


class FeatureExtractor(torch.nn.Module):
    """EfficientNet feature extractor"""

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.feature_extractor = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

    def forward(self, x):
        return self.feature_extractor(x)
