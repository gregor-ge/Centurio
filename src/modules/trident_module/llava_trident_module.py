import hydra

from trident import TridentModule
import torch
from typing import Union, Optional, cast, Any
from transformers import BatchEncoding

class LlavaTridentModule(TridentModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strict_loading = False
        self.freeze_vit = kwargs["model"].get("freeze_vit", False)
        # self.automatic_optimization = False
        # self.gradient_clip_val = 1.0

    def on_save_checkpoint(self, checkpoint) -> None:
        filter_keys = ["language_model"]
        if self.freeze_vit:
            filter_keys.append("vision_tower")
        checkpoint['state_dict'] = {k:v for k,v in checkpoint['state_dict'].items() if not any(f in k for f in filter_keys)}
        pass

    def configure_optimizers(self):
        """Prepares optimizer and scheduler."""
        weight_decay = getattr(self.hparams.optimizer, "weight_decay", 0)
        
        lr = self.hparams.optimizer.lr
        lora_lr = getattr(self.hparams.optimizer, "lora_lr", lr)
        if hasattr(self.hparams.optimizer, "lora_lr"):
            delattr(self.hparams.optimizer, "lora_lr")
        def is_lora(param):
            return "lora" in param
        def is_vit(param):
            return "vision_tower" in param

        param_optimizer = list(self.named_parameters())
        no_decay = ["LayerNorm.bias", "LayerNorm.weight"]
        parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if is_lora(n)
                ],
                "weight_decay": weight_decay,
                "lr": lora_lr
            },
            # {
            #     "params": [
            #         p for n, p in param_optimizer if any(nd in n for nd in no_decay) and is_lora(n)
            #     ],
            #     "weight_decay": 0.0,
            #     "lr": lora_lr
            # },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and not is_lora(n) and not is_vit(n)
                ],
                "weight_decay": weight_decay,
                "lr": lr
            },
            # {
            #     "params": [
            #         p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not is_lora(n) and not is_vit(n)
            #     ],
            #     "weight_decay": 0.0,
            #     "lr": lr
            # },
        ]

        if not self.freeze_vit:
            vit_lr = getattr(self.hparams.optimizer, "vit_lr", lr)
            if hasattr(self.hparams.optimizer, "vit_lr"):
                delattr(self.hparams.optimizer, "vit_lr")
            vit_weight_decay = getattr(self.hparams.optimizer, "vit_weight_decay", weight_decay)
            if hasattr(self.hparams.optimizer, "vit_weight_decay"):
                delattr(self.hparams.optimizer, "vit_weight_decay")
            parameters.extend([
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not "norm" in n and is_vit(n)
                ],
                "weight_decay": vit_weight_decay,
                "lr": vit_lr
            },
            {
                "params": [
                    p for n, p in param_optimizer if "norm" in n and is_vit(n)
                ],
                "weight_decay": 0.0,
                "lr": vit_lr
            },
            ])


        optimizer = hydra.utils.instantiate(self.hparams.optimizer, parameters)
        if scheduler_cfg := getattr(self.hparams, "scheduler"):
            scheduler = self.configure_scheduler(optimizer, scheduler_cfg)
            return [optimizer], [scheduler]
        return [optimizer]

    def training_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Comprises training step of your model which takes a forward pass.

        **Notes:**
            If you want to extend `training_step`, add a `on_train_batch_end` method via overrides.
            See: Pytorch-Lightning's `on_train_batch_end <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end>`_

        Args:
            batch: typically comprising input_ids, attention_mask, and position_ids
            batch_idx: variable used internally by pytorch-lightning

        Returns:
            Union[dict[str, Any], ModelOutput]: model output that must have 'loss' as attr or key
        """
        outputs = self(batch)
        self.log("train/loss", outputs["loss"])
        for key in outputs.keys():
            if "metric" in key:
                self.log(f"train/{key}", outputs[key])
        return outputs
