import inspect
import logging
import os
import warnings
from functools import partial

import timm
import torch
from accelerate.utils import get_balanced_memory
from lightning import LightningModule
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, LlavaConfig, AutoConfig, AutoModelForCausalLM, T5ForConditionalGeneration, \
    Idefics2Config, AutoModelForVision2Seq, LlamaModel, AutoModel

from accelerate import dispatch_model, infer_auto_device_map
from src.modules.modeling.llava import LlavaForConditionalGeneration, LlavaForSeqToSeqGeneration, \
    LlavaIdefics2ChimeraForConditionalGeneration, LlavaAIMForConditionalGeneration
from src.tasks.vllm.data import get_tokenizer_llava, IMAGE_TOKEN


class LlavaModule(LightningModule):
    def __init__(self,
                 checkpoint=None,
                 inference_only=False,
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
                 lora_dropout=0.0,
                 lora_checkpoint=None,
                 merge_lora_checkpoint=False,
                 lora_dora=False,
                 lora_init=True,
                 lora_rs=False,
                 adapter_config={},
                 adapter_type="mlp",
                 overwrite_size=-1,
                 train_pos_embed=False,
                 cast_fp32=None,
                 **kwargs
        ):
        super().__init__()

        kwargs = {"device_map": "auto"}
        kwargs['torch_dtype'] = torch.bfloat16

        if not inference_only:
            local_rank = int(os.environ.get('SLURM_LOCALID', os.environ.get("LOCAL_RANK", 0)))
            logging.info(f'Loading model at rank {local_rank}')
            kwargs["device_map"] = {"": local_rank}
        else:
            kwargs["device_map"] = "balanced_low_0"

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


        is_seq2seq =  ("t5" in lm_pretrained or "aya-101" in lm_pretrained or "t0" in lm_pretrained)

        llava_class = LlavaForConditionalGeneration if not is_seq2seq else LlavaForSeqToSeqGeneration

        if checkpoint is not None:
            self.model = llava_class.from_pretrained(checkpoint, timm_model=vit_pretrained, **kwargs)
        else:
            attention = "flash_attention_2" if use_flash_attn else "sdpa"
            if is_seq2seq or (not use_flash_attn and any(model in lm_pretrained for model in {"stabilityai", "gemma-2-"})):
                attention = "eager"
            kwargs["attn_implementation"] = attention
            text_config = AutoConfig.from_pretrained(lm_pretrained, trust_remote_code=True)
            config = LlavaConfig(text_config=text_config, attn_implementation=attention)
            config.timm_model = vit_pretrained
            config.vision_feature_select_strategy = "full"

            config.adapter_type = adapter_type
            # ps_queries, layers
            for k,v in adapter_config.items():
                setattr(config, f"adapter_{k}", v)

            tokenizer = get_tokenizer_llava(lm_pretrained)
            # stabilityai tokenizer does not allow for added tokens so we re-appropiate/ use one of the existing ones
            if "stabilityai" in lm_pretrained:
                config.image_token_index = tokenizer.convert_tokens_to_ids("<filename>")
            else:
                config.image_token_index = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            config.torch_dtype=torch.bfloat16
            config.delay_init = True
            # has to be set for generation() to work with cache
            if "gemma-2-" in lm_pretrained:
                config.sliding_window = text_config.sliding_window
                config.sliding_window_size = text_config.sliding_window_size
                config.head_dim = text_config.head_dim
                config.num_attention_heads = text_config.num_attention_heads
                config.num_key_value_heads = text_config.num_key_value_heads
                config.num_hidden_layers = text_config.num_hidden_layers

            self.model = llava_class(config)
            if not is_seq2seq:
                self.model.language_model = AutoModelForCausalLM.from_pretrained(lm_pretrained, trust_remote_code=True, **kwargs)
                if inference_only:
                    logging.info(self.model.language_model.hf_device_map)
            else:
                self.model.language_model = T5ForConditionalGeneration.from_pretrained(lm_pretrained, trust_remote_code=True, **kwargs)
            if overwrite_size>0:
                self.model.vision_tower = timm.create_model(
                    config.timm_model,
                    img_size=overwrite_size,
                    pretrained=True,
                    num_classes=0,
                )
            else:
                self.model.vision_tower = timm.create_model(
                    config.timm_model,
                    pretrained=True,
                    num_classes=0,
                )

            if config.text_config.vocab_size < len(tokenizer):
                self.model.language_model.resize_token_embeddings(len(tokenizer), 64)


            # https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/backbones/vision/base_vision.py#L125
            def unpack_tuple(fn):
                def wrapper(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    return result[0] if isinstance(result, tuple) or isinstance(result, list)  else result

                return wrapper

            self.model.vision_tower.forward = unpack_tuple(
                partial(
                    self.model.vision_tower.get_intermediate_layers, n={len(self.model.vision_tower.blocks) - 2}
                )
            )

        if cast_fp32 is None:
            cast_fp32 = load_8bit or load_4bit

        self.model.language_model = prepare_model_for_kbit_training(self.model.language_model,
                                            use_gradient_checkpointing=gradient_checkpoint,
                                                                    cast_fp32=cast_fp32)


        if freeze_vit:
            logging.info("Freeze ViT")
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            if overwrite_size>0 and train_pos_embed:
                self.model.vision_tower.pos_embed.requires_grad = True


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
                self.model.language_model = PeftModel.from_pretrained(self.model.language_model, lora_checkpoint, is_trainable=True)
                if merge_lora_checkpoint:
                    logging.info(f"Merging LoRA adapter {merge_lora_checkpoint}")
                    self.model.language_model = self.model.language_model.merge_and_unload()
            if not inference_only and (lora_checkpoint is None or merge_lora_checkpoint):
                logging.info("Instantiating LoRA model")
                task = "SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM" #if "bloom" in lm_pretrained or "poly" in lm_pretrained else "SEQ_2_SEQ_LM"
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules="all-linear",
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    task_type=task,
                    init_lora_weights=lora_init,
                    use_dora=lora_dora,
                    use_rslora=lora_rs
                )

                self.model.language_model = get_peft_model(self.model.language_model, config)

        if gradient_checkpoint:
            self.model.language_model.gradient_checkpointing_enable()

        if compile:
            self.model = torch.compile(self.model)

        # if inference_only:
        #     self.inference_only = True
        #     self.dispatched = False
        #     #
        #     max_memory = get_balanced_memory(
        #         self.model.vision_tower,
        #         max_memory=None,
        #         low_zero=True,
        #     )
        #     device_map = infer_auto_device_map(
        #         self.model,
        #         max_memory=max_memory,
        #     )
        #     self.model = dispatch_model(self.model, device_map)
        #     self.dispatched = True
        # else:
        #     self.inference_only = False


    def forward(self, mode="forward", split=None, dataset_name=None, **kwargs):
        if mode == "forward":
            return self.model(**kwargs)
        else:
            return self.generate(**kwargs)

    def generate(self, **kwargs):

        generate_kwargs = kwargs.get("generate_kwargs", dict())
        generate_kwargs.pop("stage", None) # added by Trident but must go

        if hasattr(self.model.language_model.config, "cache_implementation") and "cache_implementation" not in generate_kwargs:
            generate_kwargs["cache_implementation"] = self.model.language_model.config.cache_implementation

        model_inputs = {k: kwargs[k] for k in ["input_ids", "pixel_values", "attention_mask", "decoder_input_ids", "decoder_attention_mask"] if k in kwargs}
        # has to be done for Gemma2 because the Cache uses as dtype self.model.dtype which is default float32 but we need bfloat16...
        self.model.to(dtype=torch.bfloat16)
        return self.model.generate(**model_inputs, **kwargs["generate_kwargs"])

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None, cast_fp32=False):
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

    if cast_fp32:
        # cast all non INT8 parameters to fp32
        for name, param in model.named_parameters():
            if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and ("norm" in name or "lm_head" in name):
                param.data = param.data.to(torch.float32)

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