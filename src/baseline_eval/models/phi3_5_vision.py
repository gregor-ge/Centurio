from typing import Any

from transformers import AutoModelForCausalLM, AutoProcessor

from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import IMAGE_PLACEHOLDER_TOKEN, load_image


class Phi3_5_Vision(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "microsoft/Phi-3.5-vision-instruct",
        device: str = "cuda",
    ):
        self.multi_image_processor = AutoProcessor.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            num_crops=4,
        )
        self.single_image_processor = AutoProcessor.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            num_crops=16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
        )
        self.device = device

        self.generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

    def _get_processor(self, image_paths: list[str]):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if len(image_paths) == 1:
            processor = self.single_image_processor
        elif len(image_paths) > 1:
            processor = self.multi_image_processor
        else:
            raise NotImplementedError("No image paths provided")

        return processor

    def generate_text(self, prompt: str, image_paths: list[str], **kwargs) -> list[str]:
        inputs = self._prepare_model_input(prompt, image_paths)
        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=self.single_image_processor.tokenizer.eos_token_id,
            **self.generation_args,
        )
        # trim the input tokens from the generated output
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]

        processor = self._get_processor(image_paths)

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_texts

    def _prepare_model_input(self, prompt: str, image_paths: list[str]) -> Any:
        if not isinstance(image_paths, list):
            raise ValueError(f"Image paths must be a list: {type(image_paths)}")

        # count the number of IMAGE_PLACEHOLDER_TOKENs in the prompt
        num_image_placeholders = prompt.count(IMAGE_PLACEHOLDER_TOKEN)

        if num_image_placeholders > 0 and num_image_placeholders != len(image_paths):
            raise ValueError(
                f"Number of image paths ({len(image_paths)}) must match the number of IMAGE_PLACEHOLDER_TOKENs in the prompt ({num_image_placeholders})"
            )

        if num_image_placeholders == 0 and len(image_paths) == 1:
            prompt = IMAGE_PLACEHOLDER_TOKEN + "\n" + prompt
            num_image_placeholders = 1

        # replace the IMAGE_PLACEHOLDER_TOKEN with the Phi3.5 image placeholder tokens
        phi_image_placeholder_token = "<|image_{NUM}|>"
        for i in range(1, num_image_placeholders + 1):
            prompt = prompt.replace(
                IMAGE_PLACEHOLDER_TOKEN, phi_image_placeholder_token.format(NUM=i), 1
            )
        messages = [
            {"role": "user", "content": prompt},
        ]

        processor = self._get_processor(image_paths)

        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        images = [load_image(image_path) for image_path in image_paths]

        inputs = processor(prompt, images, return_tensors="pt").to(self.device)

        return inputs
