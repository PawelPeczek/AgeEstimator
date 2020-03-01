from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

from .model import resnet34

torch.backends.cudnn.deterministic = True


class AgeEstimator:

    __PREDICTED_AGE_SHIFT = 16

    @classmethod
    def initialize(cls,
                   weights_path: str,
                   min_prediction_probability: float = 0.4,
                   use_gpu: bool = False
                   ) -> AgeEstimator:
        device = torch.device("cuda" if use_gpu else "cpu")
        model = resnet34(num_classes=55, grayscale=False)
        model.load_state_dict(
            torch.load(weights_path, map_location=device)
        )
        return cls(
            model=model,
            device=device,
            min_prediction_probability=min_prediction_probability
        )

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 min_prediction_probability: float):
        self.__model = model
        self.__model.eval()
        self.__device = device
        self.__min_prediction_probability = min_prediction_probability
        self.__normalization_transformation = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop((120, 120)),
            transforms.ToTensor()]
        )

    def estimate_age(self,
                     image: np.ndarray,
                     image_in_rgb_mode: bool = False
                     ) -> Optional[int]:
        image = self.__normalize_input(
            image=image,
            image_in_rgb_mode=image_in_rgb_mode
        )
        with torch.no_grad():
            _, probabilities = self.__model(image)
        probabilities = probabilities.cpu()
        predicted_class = torch.argmax(probabilities, 1).item()
        return predicted_class + AgeEstimator.__PREDICTED_AGE_SHIFT

    def __normalize_input(self,
                          image: np.ndarray,
                          image_in_rgb_mode: bool = False
                          ) -> torch.Tensor:
        if not image_in_rgb_mode:
            image = image[:, :, ::-1]
        image = self.__normalization_transformation(image)
        image = image.to(self.__device)
        return image.unsqueeze(0)
