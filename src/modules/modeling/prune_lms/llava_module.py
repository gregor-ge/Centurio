import inspect
import logging
import warnings
from functools import partial

import timm
import torch
from lightning import LightningModule
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, LlavaConfig, AutoConfig, AutoModelForCausalLM

from src.modules.modeling.prune_lms.llava import LlavaForConditionalGeneration
from src.tasks.vllm.data import get_tokenizer_llava, IMAGE_TOKEN
from src.modules.modeling.prune_lms.modeling_stablelm import StableLmForCausalLM

class LlavaModule(LightningModule):
    def __init__(self,
                 checkpoint=None,
                 lm_pretrained="stabilityai/stablelm-2-zephyr-1_6b",
                 vit_pretrained="vit_large_patch14_clip_224.openai",
                 train_checkpoint=None,
                 load_8bit=False,
                 load_4bit=True,
                 use_flash_attn=False,
                 freeze_vit=True,
                 freeze_lm=True,
                 compile=False,
                 gradient_checkpoint=False,
                 use_lora=False,
                 lora_alpha=64,
                 lora_r=32,
                 lora_bias="none",
                 lora_dropout=0.05,
                 lora_checkpoint=None,
                 adapter_config={},
                 adapter_type="mlp",
                 prune_layer=2,
                 prune_keep=0.5,
                 prune_mode="attention",
                 num_query_tokens=0,
                 **kwargs
        ):
        super().__init__()

        kwargs = {"device_map": "auto"}

        kwargs['torch_dtype'] = torch.bfloat16

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs.update(dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4'
            ))

        if use_flash_attn:
            kwargs['attn_implementation'] = 'flash_attention_2'


        if checkpoint is not None:
            self.model = LlavaForConditionalGeneration.from_pretrained(checkpoint, timm_model=vit_pretrained, **kwargs)
        else:
            attention = "flash_attention_2" if use_flash_attn else "sdpa"
            if not use_flash_attn and any(model in lm_pretrained for model in {"stabilityai"}):
                attention = "eager"
            text_config = AutoConfig.from_pretrained(lm_pretrained)

            text_config.do_prune = True
            text_config.prune_layer = prune_layer
            text_config.prune_keep = prune_keep
            text_config.prune_mode = prune_mode

            config = LlavaConfig(text_config=text_config, attn_implementation=attention)
            config.timm_model = vit_pretrained
            config.vision_feature_select_strategy = "full"
            config.num_query_tokens = num_query_tokens

            config.adapter_type = adapter_type
            # ps_queries, layers
            for k,v in adapter_config.items():
                setattr(config, f"adapter_{k}", v)

            tokenizer = get_tokenizer_llava(lm_pretrained)
            config.image_token_index = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            config.torch_dtype=torch.bfloat16

            self.model = LlavaForConditionalGeneration(config, )
            if "stablelm" in lm_pretrained:
                self.model.language_model = StableLmForCausalLM.from_pretrained(lm_pretrained, **kwargs)
            self.model.language_model.config.do_prune = True
            self.model.language_model.config.prune_layer = prune_layer
            self.model.language_model.config.prune_keep = prune_keep
            self.model.language_model.config.prune_mode = prune_mode

            # self.model.language_model = AutoModelForCausalLM.from_pretrained(lm_pretrained, **kwargs)
            self.model.vision_tower = timm.create_model(
                config.timm_model,
                pretrained=True,
                num_classes=0,
            )

            # https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/backbones/vision/base_vision.py#L125
            def unpack_tuple(fn):
                def wrapper(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    return result[0] if isinstance(result, tuple) else result

                return wrapper

            self.model.vision_tower.forward = unpack_tuple(
                partial(
                    self.model.vision_tower.get_intermediate_layers, n={len(self.model.vision_tower.blocks) - 2}
                )
            )

        self.model.language_model = prepare_model_for_kbit_training(self.model.language_model,
                                                use_gradient_checkpointing=gradient_checkpoint)

        if freeze_vit:
            logging.info("Freeze ViT")
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False

        if freeze_lm:
            logging.info("Freeze LLM")
            for param in self.model.language_model.parameters():
                param.requires_grad = False

        if train_checkpoint:
            logging.info(f"Loading training checkpoint {train_checkpoint}")
            checkpoint = torch.load(train_checkpoint, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
                checkpoint = {k.replace("model.model.", "model."): v for k, v in checkpoint.items()}
            if "module" in checkpoint:
                checkpoint = checkpoint["module"]
                checkpoint = {k.replace("model.model.", "model."): v for k, v in checkpoint.items()
                              if "quant" not in k and "absmax" not in k}  # deepspeed saves some quant params, lets remove them to be sure

            missing, unexpected = self.load_state_dict(checkpoint, strict=False)
            logging.info(f"Unexpected weights from training checkpoint: {unexpected}")

        if use_lora:
            logging.info("Using LoRA")
            if lora_checkpoint:
                logging.info(f"Loading LoRA adapter {lora_checkpoint}")
                self.model.language_model = PeftModel.from_pretrained(self.model.language_model, lora_checkpoint)
            else:
                task = "CAUSAL_LM" #if "bloom" in lm_pretrained or "poly" in lm_pretrained else "SEQ_2_SEQ_LM"
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules="all-linear",
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    task_type=task
                )

                self.model.language_model = get_peft_model(self.model.language_model, config)

        if gradient_checkpoint:
            self.model.language_model.gradient_checkpointing_enable()

        # if compile:  # compile currently does not work with gradient checkpoint 2.0.1
        #     self.model = torch.compile(self.model)


    def forward(self, mode="forward", split=None, dataset_name=None, **kwargs):
        if mode == "forward":
            return self.model(**kwargs)
        else:
            return self.generate(**kwargs)

    def generate(self, **kwargs):
        generate_kwargs = kwargs.get("generate_kwargs", dict())
        generate_kwargs.pop("stage", None) # added by Trident but must go
        return self.model.generate(pixel_values=kwargs["pixel_values"], input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"], **kwargs["generate_kwargs"])




def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    r"""
    Note this method only works for `transformers` models.

    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
        use_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        gradient_checkpointing_kwargs (`dict`, *optional*, defaults to `None`):
            Keyword arguments to pass to the gradient checkpointing function, please refer to the documentation of
            `torch.utils.checkpoint.checkpoint` for more details about the arguments that you can pass to that method.
            Note this is only available in the latest transformers versions (> 4.34.1).
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    # for name, param in model.named_parameters():
    #     # freeze base model's layers
    #     param.requires_grad = False

    # if not is_gptq_quantized:
    #     # cast all non INT8 parameters to fp32
    #     for param in model.parameters():
    #         if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #             param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model