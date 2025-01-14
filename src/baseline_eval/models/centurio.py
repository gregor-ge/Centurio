import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src.baseline_eval.data_utils import IMAGE_PLACEHOLDER_TOKEN
from src.baseline_eval.models.baseline_model import BaselineModel


class Centurio(BaselineModel):
    def __init__(self, hf_model_id: str, device: str = "cuda"):
        if hf_model_id not in ["WueNLP/centurio_aya", "WueNLP/centurio_qwen"]:
            raise ValueError("Model ID not supported")
        token = os.environ.get("HF_TOKEN", None)

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                token=token,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoProcessor.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            use_fast=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=token,
            padding_side="right" if hf_model_id == "WueNLP/centurio_qwen" else "left",
        )

        self.device = device
        self.hf_model_id = hf_model_id

        self.generation_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

        if hf_model_id == "WueNLP/centurio_qwen":
            self.generation_kwargs = {
                **self.generation_kwargs,
                "pad_token_id": 151643,
                "min_new_tokens": 1,
            }

    def apply_chat_template(self, prompt: str):
        if self.hf_model_id == "WueNLP/centurio_qwen":
            prompt_template = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{PROMPT}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            prompt_template = (
                "<BOS_TOKEN>"
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>\n{PROMPT}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            )

        return prompt_template.format(PROMPT=prompt)

    def _prepare_model_inputs(self, prompt: str, image_paths: list[str]) -> dict:
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        num_image_placeholders = prompt.count(IMAGE_PLACEHOLDER_TOKEN)

        if num_image_placeholders == 0 and len(image_paths) == 1:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt
            num_image_placeholders = 1

        if num_image_placeholders != len(image_paths):
            raise ValueError(
                f"Number of image paths ({len(image_paths)}) must match the number "
                f"of {IMAGE_PLACEHOLDER_TOKEN} in the prompt ({num_image_placeholders})"
            )

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = self.tokenizer(
            images=images,
            text=self.apply_chat_template(prompt),
            return_tensors="pt",
        ).to(self.device, torch.bfloat16)

        return inputs

    def generate_text(self, prompt: str, image_paths: list[str], **kwargs) -> list[str]:
        model_inputs = self._prepare_model_inputs(prompt, image_paths)

        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **self.generation_kwargs)
            outputs = outputs[:, model_inputs["input_ids"].shape[1] :]

        return [
            s.strip()
            for s in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
