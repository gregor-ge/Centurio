import torch
from transformers import AutoModel, AutoTokenizer

from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import load_image, get_prompt_parts


class MiniCPM_V_2_6(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "openbmb/MiniCPM-V-2_6",
        device: str = "cuda",
    ):
        self.model = AutoModel.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
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
        messages = self._prepare_messages(prompt, image_paths)
        answer = self.model.chat(
            image=None,
            msgs=messages,
            tokenizer=self.tokenizer,
            **self.generation_args,
        )
        return [answer]

    def _prepare_messages(self, prompt: str, image_paths: list[str]) -> list[str]:
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

        images = [load_image(image_path) for image_path in image_paths]
        content = []
        for i, prompt_part in enumerate(prompt_parts):
            if prompt_part == "":
                content.append(images[i])
            else:
                if not (prompt_part.startswith(" ") or prompt_part.endswith(" ")):
                    prompt_part += " "
                content.append(prompt_part)

        messages = [{"role": "user", "content": content}]

        return messages
