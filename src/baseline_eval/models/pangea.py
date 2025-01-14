import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor

from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import (
    load_image,
    IMAGE_PLACEHOLDER_TOKEN,
)


class Pangea(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "neulab/Pangea-7B-hf",
        device: str = "cuda",
    ):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(device)
        self.processor = AutoProcessor.from_pretrained("neulab/Pangea-7B-hf")
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

        self.device = device

        self.generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

    def generate_text(self, prompt: str, image_paths: list[str], **kwargs) -> list[str]:
        model_inputs = self._prepare_model_inputs(prompt, image_paths)
        generated_ids = self.model.generate(
            **model_inputs,
            **self.generation_args,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        generated_ids_trimmed = generated_ids[:, model_inputs["input_ids"].shape[1] :]

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_texts

    def _prepare_model_inputs(self, prompt: str, image_paths: list[str]) -> dict:
        if not isinstance(image_paths, list):
            raise ValueError(f"Image paths must be a list: {type(image_paths)}")

        img_token_cnt = prompt.count(IMAGE_PLACEHOLDER_TOKEN)
        if img_token_cnt == 0 and len(image_paths) == 1:
            prompt = f"{IMAGE_PLACEHOLDER_TOKEN}\n{prompt}"
            img_token_cnt = 1

        if not img_token_cnt == len(image_paths):
            raise ValueError(
                f"Number of image paths ({len(image_paths)}) must match the number "
                f"of {IMAGE_PLACEHOLDER_TOKEN} in the prompt ({img_token_cnt})"
            )

        pangea_image_token = "<image>"
        user_prompt = prompt.replace(IMAGE_PLACEHOLDER_TOKEN, pangea_image_token)

        text_input = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        images = [load_image(image_path) for image_path in image_paths]

        model_inputs = self.processor(
            images=images, text=text_input, return_tensors="pt"
        ).to(self.device, torch.float16)

        return model_inputs
