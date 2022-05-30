import timm
from torch import nn

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name, pretrained, trainable, config,
    ):
        super().__init__()
        self.model_name = config['image_encoder']['model_name']
        self.pretrained = config['global']['pretrained']
        self.trainable = config['global']['trainable']

        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)