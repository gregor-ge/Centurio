from typing import Any

from transformers import MllamaForConditionalGeneration, AutoProcessor

import torch
from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import IMAGE_PLACEHOLDER_TOKEN, load_image


class Llama3_2_Vision(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        device: str = "cuda",
    ):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            hf_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(hf_model_id)

        self.device = device

        self.generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

    def generate_text(self, prompt: str, image_paths: list[str], **kwargs) -> list[str]:
        inputs = self._prepare_model_input(prompt, image_paths)
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_args,
        )
        # trim the input tokens from the generated output
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_texts

    def _prepare_model_input(
        self, prompt: str, image_paths: list[str]
    ) -> dict[str, Any]:
        if not image_paths:
            raise ValueError("No image paths provided")
        elif not isinstance(image_paths, list):
            raise ValueError(f"Image paths must be a list: {type(image_paths)}")
        elif len(image_paths) > 1:
            raise NotImplementedError("Only one image path is supported")

        image = load_image(image_paths[0])

        if prompt.count(IMAGE_PLACEHOLDER_TOKEN) > 1:
            raise NotImplementedError("Only one image placeholder token is supported")

        prompt.replace(IMAGE_PLACEHOLDER_TOKEN, "")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        return inputs
