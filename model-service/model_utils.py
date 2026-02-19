import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io

CLASS_NAMES = ["benign", "malignant",]

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.AdaptiveAvgPool2d((6, 6))
                                     )
        
        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 2)
                                       )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 6 * 6 * 256)
        x = self.classifier(x)
        return x

# same transforms you used in training for test/validation
IMAGE_SIZE = (227, 227)

transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image_input) -> torch.Tensor:
    """
    Preprocess an image for inference.

    Accepts bytes (uploaded file), a file path (str), or a PIL.Image.
    Returns a [1, 3, H, W] float tensor.
    """
    if isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        raise TypeError("image_input must be bytes, str (file path), or PIL.Image")

    image = image.convert("RGB")
    x = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return x