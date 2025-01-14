from abc import ABC, abstractmethod


class BaselineModel(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, image_paths: list[str], **kwargs) -> list[str]:
        pass
