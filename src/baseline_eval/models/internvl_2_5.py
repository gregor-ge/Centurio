import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from src.baseline_eval.models.baseline_model import BaselineModel
from src.baseline_eval.data_utils import IMAGE_PLACEHOLDER_TOKEN

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class Intern_V_2_5(BaselineModel):
    def __init__(
        self,
        hf_model_id: str = "OpenGVLab/InternVL2_5-8B",
        device: str = "cuda",
    ):
        if not hf_model_id.startswith("OpenGVLab/InternVL2_5-"):
            raise ValueError(f"Invalid model ID: {hf_model_id}")
        self.model = (
            AutoModel.from_pretrained(
                hf_model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id, trust_remote_code=True, use_fast=False
        )

        self.device = device

        self.generation_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }

    def generate_text(self, prompt: str, image_paths: list[str], **kwargs) -> list[str]:
        prompt, pixel_values, num_patches_list = self._prepare_model_inputs(
            prompt, image_paths
        )
        answer = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            self.generation_kwargs,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

        return [answer]

    def _prepare_model_inputs(
        self, prompt: str, image_paths: list[str]
    ) -> tuple[str, torch.Tensor, list]:
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

        if num_image_placeholders > 1:
            # according to the authors, it improves the performance to number
            # the images if there are more than one
            for i in range(1, num_image_placeholders + 1):
                prompt = prompt.replace(
                    IMAGE_PLACEHOLDER_TOKEN, f"Image-{i}: <image>", 1
                )
        else:
            prompt = prompt.replace(IMAGE_PLACEHOLDER_TOKEN, "<image>", 1)

        pixel_values = []
        num_patches_list = []
        for img in image_paths:
            pixel_values1 = load_image(img, max_num=12).to(torch.bfloat16).cuda()
            pixel_values.append(pixel_values1)
            num_patches_list.append(pixel_values1.size(0))
        pixel_values = torch.cat(pixel_values, dim=0)

        return prompt, pixel_values, num_patches_list
