from typing import Any

from transformers import AutoModelForCausalLM, AutoProcessor

from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import IMAGE_PLACEHOLDER_TOKEN, load_image


class LlavaOneVision(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
        device: str = "cuda",
    ):
        raise NotImplementedError(f"Model ID {hf_model_id} not yet implemented")
