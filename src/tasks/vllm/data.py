import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import logging
import numpy as np
import timm
import torchvision.transforms.functional as TVF
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from transformers import Blip2Processor, AutoTokenizer, PreTrainedTokenizerBase, AddedToken, AutoImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

from transformers.utils import logging as t_logging
# https://stackoverflow.com/questions/74748116/huggingface-automodelforcasuallm-decoder-only-architecture-warning-even-after
t_logging.set_verbosity_error()

IMAGE_TOKEN = "<image_placeholder>" # "<image>"

def get_tokenizer_llava(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    image_token = IMAGE_TOKEN #
    if all(model not in model_name_or_path for model in {"stabilityai"}):
        tokenizer.add_tokens(AddedToken(image_token, special=True, normalized=False), special_tokens=True)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def flatten_tensor_structure(structure):
    flattened = []
    if not isinstance(structure, list):
        flattened.append(structure)
    else:
        for item in structure:
            flattened.extend(flatten_tensor_structure(item))

    return flattened

@dataclass
class DataCollatorForVisualCLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    padding_side: str = "left"
    image_processing: str = ""

    def __call__(self, features):
        if isinstance(self.image_processing, str) and "idefics" in self.image_processing:
            self.image_processing = AutoImageProcessor.from_pretrained(self.image_processing)

        self.tokenizer.padding_side = self.padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token='[PAD]'

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = {key: [example[key] for example in features] for key in features[0].keys()}

        text_features = {k: features[k] for k in ["input_ids", "attention_mask"]}
        batch = self.tokenizer.pad(
            text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if "decoder_input_ids" in features:
            text_features = {k: features[f"decoder_{k}"] for k in ["input_ids", "attention_mask"]}
            decoder_batch = self.tokenizer.pad(
                text_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            for k,v in decoder_batch.items():
                batch[f"decoder_{k}"] = v

        batch["labels"] = torch.tensor(features["labels"])

        if "pixel_values" in features:

            pixel_values = features["pixel_values"]
            pixel_values = flatten_tensor_structure(pixel_values)
            # Filter the Nones added for image-free examples; they have to be added for datasets to work
            pixel_values = [pv for pv in pixel_values if pv is not None]

            if len(pixel_values) == 0:
                batch["pixel_values"] = None
            else:
                if not isinstance(self.image_processing, str):
                    pixel_values, pixel_mask = self.image_processing.pad([pixel_values], return_tensors="pt")
                    batch["pixel_attention_mask"] = torch.tensor(pixel_mask[0])
                    batch["pixel_values"] = torch.tensor(pixel_values[0])
                else:
                    batch["pixel_values"] = torch.stack(pixel_values)

        # add everything else as is
        for k, v in features.items():
            if k not in batch:
                batch[k] = v

        batch = batch.data  # BatchEncoding to dict
        return batch


@dataclass
class DataCollatorForVisualDPO:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    padding_side: str = "left"
    image_processing: str = ""

    def __call__(self, features):
        if isinstance(self.image_processing, str) and "idefics" in self.image_processing:
            self.image_processing = AutoImageProcessor.from_pretrained(self.image_processing)

        self.tokenizer.padding_side = self.padding_side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token='[PAD]'

        chosen_labels = [feature["chosen_labels"] for feature in features] if "chosen_labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if chosen_labels is not None:
            max_label_length = max(len(l) for l in chosen_labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["chosen_labels"]))
                if isinstance(feature["chosen_labels"], list):
                    feature["chosen_labels"] = (
                        feature["chosen_labels"] + remainder if padding_side == "right" else remainder + feature["chosen_labels"]
                    )
                elif padding_side == "right":
                    feature["chosen_labels"] = np.concatenate([feature["chosen_labels"], remainder]).astype(np.int64)
                else:
                    feature["chosen_labels"] = np.concatenate([remainder, feature["chosen_labels"]]).astype(np.int64)


        rejected_labels = [feature["rejected_labels"] for feature in features] if "rejected_labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if rejected_labels is not None:
            max_label_length = max(len(l) for l in rejected_labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["rejected_labels"]))
                if isinstance(feature["rejected_labels"], list):
                    feature["rejected_labels"] = (
                        feature["rejected_labels"] + remainder if padding_side == "right" else remainder + feature["rejected_labels"]
                    )
                elif padding_side == "right":
                    feature["rejected_labels"] = np.concatenate([feature["rejected_labels"], remainder]).astype(np.int64)
                else:
                    feature["rejected_labels"] = np.concatenate([remainder, feature["rejected_labels"]]).astype(np.int64)

        features = {key: [example[key] for example in features] for key in features[0].keys()}

        chosen_text_features = {k: features["chosen_"+k] for k in ["input_ids", "attention_mask"]}
        rejected_text_features = {k: features["rejected_"+k] for k in ["input_ids", "attention_mask"]}
        chosen_batch = self.tokenizer.pad(
            chosen_text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        rejected_batch = self.tokenizer.pad(
            rejected_text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = dict()

        for k, v in chosen_batch.items():
            batch[f"chosen_{k}"] = v
        for k, v in rejected_batch.items():
            batch[f"rejected_{k}"] = v

        batch["chosen_labels"] = torch.tensor(features["chosen_labels"])
        batch["rejected_labels"] = torch.tensor(features["rejected_labels"])


        if "pixel_values" in features:

            pixel_values = features["pixel_values"]
            assert len(features["chosen_input_ids"]) == len(pixel_values)
            # Filter the Nones added for image-free examples; they have to be added for datasets to work
            pixel_values = [pv for pv in pixel_values if pv is not None]

            if len(pixel_values) == 0:
                batch["pixel_values"] = None
            else:
                if isinstance(features["pixel_values"][0], list):
                    batch["pixel_values"] = torch.stack([torch.stack(t) for t in pixel_values])
                    batch["pixel_values"] = batch["pixel_values"].flatten(0, 1)
                else:
                    if not isinstance(self.image_processing, str):
                        pixel_values, pixel_mask = self.image_processing.pad([pixel_values], return_tensors="pt")
                        batch["pixel_attention_mask"] = torch.tensor(pixel_mask[0])
                        batch["pixel_values"] = torch.tensor(pixel_values[0])
                    else:
                        batch["pixel_values"] = torch.stack(pixel_values)

        # add everything else as is
        for k, v in features.items():
            if k not in batch:
                batch[k] = v

        return batch


text_templates = {
# stabilityai tokenizer does not allow for added tokens so we re-appropiate/ use one of the existing ones
    "stabilityai": ("", "<|user|>\n<filename>\n{}<|endoftext|>\n", "<|user|>\n{}<|endoftext|>\n", "<|user|>\n{}<|endoftext|>\n", "<|assistant|>\n{}<|endoftext|>", "<|assistant|>\n"),
    "chatml": ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
               "<|im_start|>user\n"+IMAGE_TOKEN+"\n{}<|im_end|>\n",
               "<|im_start|>user\n{}<|im_end|>\n",
               "<|im_start|>user\n{}<|im_end|>\n",
               "<|im_start|>assistant\n{}<|im_end|>",
               "<|im_start|>assistant\n"),
    "mistral": ("", "<s> [INST] "+IMAGE_TOKEN+"\n{} [/INST]",  "<s> [INST] {} [/INST]", " [INST] {} [/INST]", "{}</s>", ""),
    "mistral2": ("", "<s>[INST]"+IMAGE_TOKEN+"\n{}[/INST]",  "<s>[INST]{}[/INST]", " [INST]{}[/INST]", "{}</s>", ""),
    "t5": ("", #A chat between a user and an AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        "USER: "+IMAGE_TOKEN+"\n{}</s>", "USER: {}</s>", "USER: {}</s>", "ASSISTANT: {}</s>", "ASSISTANT:"),
    "llama3": ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.",
               "<|start_header_id|>user<|end_header_id|>\n\n"+IMAGE_TOKEN+"\n{}<|eot_id|>",
               "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
               "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
               "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
               "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "phi3": ("<s>",
             "<|user|>\n"+IMAGE_TOKEN+"\n{} <|end|>\n",
             "<|user|>\n{} <|end|>\n",
             "<|user|>\n{} <|end|>\n",
             "<|assistant|>\n{} <|end|>\n",
             "<|assistant|>\n",
    ),
    "phi3.5": ("<s><|system|>\nYou are a helpful AI assistant.<|end|>",
             "<|user|>\n"+IMAGE_TOKEN+"\n{}<|end|>\n",
             "<|user|>\n{}<|end|>\n",
             "<|user|>\n{}<|end|>\n",
             "<|assistant|>\n{}<|end|>\n",
             "<|assistant|>\n",
    ),
    "gemma": ("",
              "<bos><start_of_turn>user\n"+IMAGE_TOKEN+"{}<end_of_turn>\n",
              "<bos><start_of_turn>user\n{}<end_of_turn>\n",
              "<start_of_turn>user\n{}<end_of_turn>\n",
              "<start_of_turn>model\n{}<end_of_turn>\n",
              "<start_of_turn>model\n",
              ),
    "aya": ("",
              "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"+IMAGE_TOKEN+"{}<|END_OF_TURN_TOKEN|>",
              "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
              "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
              "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{}<|END_OF_TURN_TOKEN|>",
              "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
              ),
    "default": ("", "USER: "+IMAGE_TOKEN+"\n{}\n", "USER: {}\n", "USER: {}\n", "ASSISTANT: {}</s>", "ASSISTANT:")
}


class ImageTextProcess:
    def __init__(self, text_process, image_process):
        self.text_process = text_process
        self.image_process = image_process

    def __call__(self, example):
        # return {
        #     "input_ids": [[5]],
        #     "attention_mask": [[1]],
        #     "labels": [[5]]
        # }

        image_example = self.image_process(example)
        text_example = self.text_process(image_example)
        image_example.update(text_example)
        return image_example


class ProcessConversation:
    def __init__(self, pretrained_model, context_column="context", target_column="target", template="{}",
                 target2str=None, text_target_column=None, max_len=8192, image_tokens=-1, **kwargs):
        self.context_column = context_column
        self.target_column = target_column
        self.template = template
        self.tokenizer = get_tokenizer_llava(pretrained_model)
        self.target2str = target2str
        self.text_target_column = text_target_column
        self.max_len = max_len
        self.image_tokens = image_tokens
        # we only count "text" tokens for this.
        # if image_tokens > 0:
        #     self.max_len += image_tokens
        self.is_seq2seq = ("t5" in pretrained_model or "aya-101" in pretrained_model or "t0" in pretrained_model)
        if "stabilityai" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["stabilityai"]
        elif "Qwen" in pretrained_model or "leo" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["chatml"]
        elif "Mistral-Nemo" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["mistral2"]
        elif "mistral" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["mistral"]
        elif "t5" in pretrained_model or "t0" in pretrained_model or "aya-101" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["t5"]
        elif "Llama-3" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["llama3"]
        elif "Phi-3.5" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["phi3.5"]
        elif "Phi-3" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["phi3"]
        elif "gemma" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["gemma"]
        elif "aya" in pretrained_model:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["aya"]
        else:
            self.system_prompt, self.template_user_img, self.template_user_start, self.template_user, self.template_assistant, self.template_assistant_open = text_templates["default"]

    def __call__(self, example):
        if self.is_seq2seq:
            return self.process_seq2seq(example)
        else:
            return self.process_decoder(example)

    def process_decoder(self, example):
        final_inputs = []
        all_inputs = example[self.context_column]
        all_targets = example[self.target_column]
        max_lens = []
        for i in range(len(example[self.context_column])):
            inputs = all_inputs[i]
            if isinstance(inputs, str):
                inputs = [inputs]
            inputs = [self.template.format(x) for x in inputs]

            targets = all_targets[i]
            if isinstance(targets, str):
                targets = [targets]

            targets = [x if self.target2str is None or not x else self.target2str[x] for x in targets]
            # check if example has no image (== image_id is empty string/ first entry in image_id list is empty string)
            example_img = example["image_id"][i] if not isinstance(example["image_id"][i], list) else example["image_id"][i][0]
            template = self.template_user_img if (example_img != "" and IMAGE_TOKEN not in inputs[0]) else self.template_user_start
            user = self.system_prompt+template.format(inputs[0])
            if self.image_tokens > 0:
                user = user.replace(IMAGE_TOKEN, "".join([IMAGE_TOKEN]*self.image_tokens))
                max_len = self.max_len + user.count(IMAGE_TOKEN)
            else:
                max_len = self.max_len
            final_input = self.tokenizer(user, add_special_tokens=False)
            mask_idxs = list(range(len(final_input["input_ids"])))
            if targets[0] == "":
                target_input = self.tokenizer(self.template_assistant_open.format(targets[0]), add_special_tokens=False)
            else:
                target_input = self.tokenizer(self.template_assistant.format(targets[0]), add_special_tokens=False)

            for k in final_input:
                final_input[k] = final_input[k] + target_input[k]

            if len(final_input["input_ids"]) > max_len:
                print(f"WARNING too long sequence at first dialog turn: ''{inputs[0]} | {targets[0]}'' ")

            for i, (user, target) in enumerate(zip(inputs[1:], targets[1:])):
                if len(final_input["input_ids"]) > max_len:
                    break
                if self.image_tokens > 0:
                    user = user.replace(IMAGE_TOKEN, "".join([IMAGE_TOKEN]*self.image_tokens))
                user_input = self.tokenizer(self.template_user.format(user), add_special_tokens=False)
                mask_idxs.extend([len(final_input["input_ids"]) + i for i in range(len(user_input["input_ids"]))])
                if target == "":
                    target_input = self.tokenizer(self.template_assistant_open.format(target), add_special_tokens=False)
                else:
                    target_input = self.tokenizer(self.template_assistant.format(target), add_special_tokens=False)
                for k in final_input:
                    final_input[k] = final_input[k] + user_input[k] + target_input[k]
            labels = deepcopy(final_input["input_ids"])
            for idx in mask_idxs:
                labels[idx] = -100
            final_input["labels"] = labels
            max_lens.append(max_len)
            final_inputs.append(final_input)
        max_len = max(max_lens)
        final_inputs = {k: [f[k][:max_len] for f in final_inputs] for k in final_inputs[0].keys()}

        if self.target2str is not None and self.text_target_column is not None:
            final_inputs[self.text_target_column] = [self.target2str[e] for e in example[self.text_target_column]]

        return final_inputs

    def process_seq2seq(self, example):
        final_inputs = []
        all_inputs = example[self.context_column]
        all_targets = example[self.target_column]
        for i in range(len(example[self.context_column])):
            inputs = all_inputs[i]
            if isinstance(inputs, str):
                inputs = [inputs]
            inputs = [self.template.format(x) for x in inputs]

            targets = all_targets[i]
            if isinstance(targets, str):
                targets = [targets]

            targets = [x if self.target2str is None or not x else self.target2str[x] for x in targets]
            template = self.template_user_img if example["image_id"][i] != "" else self.template_user_start
            final_input = self.tokenizer(self.system_prompt+template.format(inputs[0]), add_special_tokens=False)
            mask_idxs = []
            if targets[0] == "":
                final_target_input = self.tokenizer(self.template_assistant_open.format(targets[0]), add_special_tokens=False)
            else:
                final_target_input = self.tokenizer(self.template_assistant.format(targets[0]), add_special_tokens=False)

            if len(final_target_input["input_ids"]) > self.max_len:
                print(f"WARNING too long sequence at first dialog turn: ''{inputs[0]} | {targets[0]}'' ")

            for i, (user, target) in enumerate(zip(inputs[1:], targets[1:])):
                if len(final_target_input["input_ids"]) > self.max_len:
                    break
                user_input = self.tokenizer(self.template_user.format(user), add_special_tokens=False)
                mask_idxs.extend([len(final_target_input["input_ids"]) + i for i in range(len(user_input["input_ids"]))])
                if target == "":
                    target_input = self.tokenizer(self.template_assistant_open.format(target), add_special_tokens=False)
                else:
                    target_input = self.tokenizer(self.template_assistant.format(target), add_special_tokens=False)

                for k in final_target_input:
                    final_target_input[k] = final_target_input[k] + user_input[k] + target_input[k]

            labels = deepcopy(final_target_input["input_ids"])
            for idx in mask_idxs:
                labels[idx] = -100
            final_input["labels"] = labels
            for k, v in final_target_input.items():
                final_input[f"decoder_{k}"] = v

            final_inputs.append(final_input)

        final_inputs = {k: [f[k][:self.max_len] for f in final_inputs] for k in final_inputs[0].keys()}

        if self.target2str is not None and self.text_target_column is not None:
            final_inputs[self.text_target_column] = [self.target2str[e] for e in example[self.text_target_column]]

        return final_inputs


#https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/backbones/vision/base_vision.py#L40C1-L49C94
@dataclass
class LetterboxPad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")

class LoadTransformImage:
    def __init__(self, image_root, processor="vit_large_patch14_clip_224.openai", target_column="image_id", extension="",
                 multi_scale=1, overwrite_size=-1, mode="square", global_image_first=True):
        self.image_root = image_root
        self.multiple_roots = not isinstance(image_root, str)
        self.target_column = target_column
        self.extension = extension
        self.multi_scale = int(multi_scale)
        self.processor = processor
        self.global_image_first = global_image_first

        if "idefics" in processor:
            self.input_size = [224, 224]
            self.transform = AutoImageProcessor.from_pretrained(processor, do_image_splitting=False)
        elif "aimv2" in processor:
            self.input_size = [336, 336]  #hardcoded for now
            self.transform = AutoImageProcessor.from_pretrained(processor)
            if self.multi_scale>1:
                self.input_size_large = (336*multi_scale, 336*multi_scale)
                self.transform_large = AutoImageProcessor.from_pretrained(processor,
                                                                          size=dict(shortest_edge=336*multi_scale),
                                                                          crop_size=dict(width=336*multi_scale, height=336*multi_scale)
                                                                          )
        else:
            config = timm.get_pretrained_cfg(processor)
            overwrite_size = int(overwrite_size)
            input_size = config.input_size[1] if overwrite_size<=0 else overwrite_size
            self.input_size = (input_size, input_size)
            self.transform = Compose([
                Resize(self.input_size, interpolation=InterpolationMode(config.interpolation)),
                ToTensor(),
                Normalize(mean=config.mean, std=config.std)
            ])
            if self.multi_scale>1:
                self.input_size_large = (input_size*multi_scale, input_size*multi_scale)
                self.transform_large = Compose([
                    Resize(self.input_size_large, interpolation=InterpolationMode(config.interpolation)),
                    ToTensor(),
                    Normalize(mean=config.mean, std=config.std)
                ])
            if mode == "letterbox":
                fill = tuple([int(x * 255) for x in config.mean])
                if self.multi_scale>1:
                    self.input_size_large = (input_size*multi_scale, input_size*multi_scale)
                    self.transform_large = Compose([
                        LetterboxPad(fill),
                        Resize(self.input_size_large, interpolation=InterpolationMode(config.interpolation)),
                        ToTensor(),
                        Normalize(mean=config.mean, std=config.std)
                    ])
                self.image_transform = Compose([
                    LetterboxPad(fill),
                    Resize(self.input_size, interpolation=InterpolationMode(config.interpolation)),
                    ToTensor(),
                    Normalize(mean=config.mean, std=config.std)])

        self.error_images = set()

    def __call__(self, examples):
        all_imgs = []
        for img_id in examples[self.target_column]:
            # Sample has no image associated with it but we have to add something to the list so the length is the batch size for datasets
            # We will remove the Nones in the collator
            if isinstance(img_id, str):
                all_imgs.append(self._load_img(img_id))
            else:
                all_imgs.append([self._load_img(img) for img in img_id])

        examples["pixel_values"] = all_imgs
        return examples

    def _load_img(self, img_id):
        if img_id == "":
            return None

        if self.multiple_roots:
            image_root = ""
            for root in self.image_root:
                if os.path.isfile(os.path.join(root, img_id + self.extension)):
                    image_root = root
                    break
        else:
            image_root = self.image_root
        image_path = os.path.join(image_root, img_id + self.extension)

        try:
            image_pil = Image.open(image_path).convert('RGB')

            if "idefics" in self.processor or "aimv2" in self.processor:
                image = self.transform(image_pil, return_tensors="pt")["pixel_values"].squeeze()
                if self.multi_scale>1:
                    image_large = self.transform_large(image_pil, return_tensors="pt")["pixel_values"].squeeze()
            else:
                image = self.transform(image_pil)  # , return_tensors="pt")["pixel_values"].squeeze()
                if self.multi_scale > 1:
                    image_large = self.transform_large(image_pil)
        except Exception as e:
            if image_path not in self.error_images:
                self.error_images.add(image_path)
                logging.warning("Failed to load image ", self.image_root, image_path)

                if len(self.error_images) > 100:
                    raise FileNotFoundError(f"Failed to load too many images {self.image_root} {self.error_images}")

            image = torch.zeros((3,) + self.input_size, dtype=torch.float32)
            if self.multi_scale > 1:
                image_large = torch.zeros((3,) + self.input_size_large, dtype=torch.float32)

        if self.multi_scale > 1:
            h, w = self.input_size
            img_large_split = [image_large[:, i * h:(i + 1) * h, j * w:(j + 1) * w] for i in range(self.multi_scale) for
                               j in range(self.multi_scale)]
            if self.global_image_first:
                return [image] + img_large_split
            else:
                return img_large_split + [image]
        else:
            return image