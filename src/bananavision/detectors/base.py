from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from bananavision.models import Detection, InferenceConfig


class Detector(ABC):
    @abstractmethod
    def predict(self, image: Image.Image, config: InferenceConfig) -> list[Detection]:
        raise NotImplementedError
