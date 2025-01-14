from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import get_prompt_parts


class Qwen2VL(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
    ):
        self.processor = AutoProcessor.from_pretrained(hf_model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
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
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_texts

    def _prepare_model_input(self, prompt: str, image_paths: list[str]) -> Any:
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

        for i in range(len(image_paths)):
            if isinstance(image_paths[i], str):
                if str(image_paths[i]).startswith("/"):
                    image_paths[i] = "file://" + str(image_paths[i])
            elif image_paths[i] is None:
                raise ValueError("Image path cannot be None")
            else:
                raise ValueError(f"Invalid image path type: {type(image_paths[i])}")

        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        for i, prompt_part in enumerate(prompt_parts):
            if prompt_part == "":
                messages[0]["content"].append(
                    {"type": "image", "image": str(image_paths[i])}
                )
            else:
                if not (prompt_part.startswith(" ") or prompt_part.endswith(" ")):
                    prompt_part += " "
                messages[0]["content"].append({"type": "text", "text": prompt_part})

        chat_messages = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=chat_messages,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        return inputs
