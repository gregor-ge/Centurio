from typing import Any
import os

from vllm import LLM
from vllm.sampling_params import SamplingParams

from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import (
    image_to_b64_url,
    get_prompt_parts,
)

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


class Pixtral(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "mistralai/Pixtral-12B-2409",
        max_img_per_msg: int = 10,
    ):
        self.model = LLM(
            model=hf_model_id,
            tokenizer_mode="mistral",
            limit_mm_per_prompt={"image": max_img_per_msg},
            max_model_len=32768 // 2,
        )
        self.sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
        )

    def generate_text(self, prompt: str, image_paths: list[str], **kwargs) -> list[str]:
        messages = self._prepare_model_messages(prompt, image_paths)

        outputs = self.model.chat(
            messages=messages,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        output_texts = [output.text for output in outputs[0].outputs]

        return output_texts

    def _prepare_model_messages(self, prompt: str, image_paths: list[str]) -> Any:
        if not isinstance(image_paths, list):
            raise ValueError(f"Image paths must be a list: {type(image_paths)}")

        if len(image_paths) == 0:
            prompt_parts = [prompt]
        else:
            prompt_parts = get_prompt_parts(prompt)
            if len(prompt_parts) != len(image_paths) + 1:
                raise ValueError(
                    f"Number of prompt parts ({len(prompt_parts)}) must be one more than the number of image paths ({len(image_paths)})"
                )

        image_urls = [image_to_b64_url(image_path) for image_path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        for i, prompt_part in enumerate(prompt_parts):
            if prompt_part == "":
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": image_urls[i]}}
                )
            else:
                if not (prompt_part.startswith(" ") or prompt_part.endswith(" ")):
                    prompt_part += " "
                messages[0]["content"].append({"type": "text", "text": prompt_part})

        return messages
