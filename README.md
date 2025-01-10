# Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2501.05122)
[![Hugging Face](https://img.shields.io/badge/Collection-%F0%9F%A4%97%20Hugging%20Face-orange)](https://huggingface.co/collections/WueNLP/centurio-677cf0ab6ddea874927a154e) 

## Release
- [2025/01/10] We released Centurio with model checkpoints and code for training & testing. Data will follow soon.


## Installation

### Standalone (with HuggingFace transformers library)


The model can be used directly through the `transformers` library with our custom code. 
Check out the model cards of our checkpoints in the [Centurio Collection on HuggingFace](https://huggingface.co/collections/WueNLP/centurio-677cf0ab6ddea874927a154e) for more details.

<details open>
<summary>Example Code</summary>

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import timm
from PIL import Image    
import requests

url = "https://upload.wikimedia.org/wikipedia/commons/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model_name = "WueNLP/centurio_qwen"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

## Appearance of images in the prompt are indicates with '<image_placeholder>'!
prompt = "<image_placeholder>\nBriefly describe the image in German."

messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # This is the system prompt used during our training.
    {"role": "user", "content": prompt}
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)

model_inputs = processor(text=[text], images=[image] return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

```
</details>

### With Trident (for training or evaluation)

We use the [trident](https://github.com/fdschmidt93/trident), a modular framework by [Fabian Schmidt](https://github.com/fdschmidt93) that combines
pytorch-lightning with hydra configs. 

```
pip install -r requirements.txt 
pip install git+https://github.com/fdschmidt93/trident.git
```

**Primer on trident:** [trident in 20 minutes](https://fdschmidt93.github.io/trident/docs/walkthrough.html)

**tl;dr:** We compose a hierarchy of configs (`/configs`) to define an experiment config (`experiment`) that defines   
(1) which datasets to use for training and testing and for the latter which metrics to use (`dataspec` and `dataspecs`),    
(2) which model to use (`module`), and   
(3) other things like optimizer, logging, checkpointing   
for a PyTorch Lightning run using our code written in `/src`.


## Training

Below is an example showing you how to use the experiment configs (`mblipv2_train.yaml` and `mblipv2_pretrain.yaml`).
Trident allows us to overwrite (nearly) all parameters specified in the configs, which we use to specify various parameters like the LLM, learning rate, etc.

For an example on how to structure the data json files, see the examples in `/data`.


<details open>
<summary>CLI Command</summary>

```
python -u -m trident.run experiment=mblipv2_train \
  run.train_data=/p/data/jsons \ # prefix path for all json
  run.image_root=/p/data/images \
  run.train_file=multilingual/combination/mblipv2_instruct_base_en.json \
  hydra.run.dir=$output \  # the output folder
  ++run.llm=microsoft/Phi-3.5-mini-instruct \
  ++run.vit_model=vit_so400m_patch14_siglip_384 \
  ++run.train_padding_side="left" \
  module.model.adapter_type="mlp" \
  module.model.load_4bit=False \
  module.model.use_flash_attn=True \
  module.model.use_lora=True \
  module.optimizer.lr=0.0001 \
  module.optimizer.weight_decay=0.0 \
  module.model.lora_r=256 module.model.lora_alpha=512 \
  ++run.max_seq_len=1024 \
  run.test_batch_size=2 run.test_num_workers=2 \
  run.train_batch_size=2 run.train_num_workers=6 \
  trainer.devices=$NUM_GPUS \  # single or multi-gpu both works out of the box
  trainer.accumulate_grad_batches=$ACCUM \
  ++run.seed=4242 \
  trainer.val_check_interval=0.25 \
  ++trainer.strategy="ddp_find_unused_parameters_true" \  # was needed for Phi 3.5 to work. Other LLMs can remove this and use the default Deepspeed Stage 2 config.
  '++logger.wandb.tags=[training,english_only]' \
```
</details>

To use the image tiling approach used for Centurio replace 
```
module.model.adapter_type="mlp" \
```

with
```
  ++run.multi_scale=2 \
  module.model.adapter_type="multiscale-pool" \
  ++module.model.adapter_config.multi_scale=2 \
```


### Evaluation

Below is an example for evaluating a model trained with the above training script on a downstream task (MAXM in this case)
by loading the checkpoint from DeepSpeed (which contains the MLP weights) and the PEFT adapter checkpoint:

<details open>
<summary>CLI Command</summary>

```
python -u -m trident.run experiment=mblipv2_test_maxm \
  run.train_data=/p/data/jsons \ # prefix path for all json
  run.xm3600_image_root=/p/data/images/maxm \
  hydra.run.dir=$output \
  ++module.model.train_checkpoint=/checkpoints/12_08_2024_09_58_16/checkpoints/0-24250.ckpt/checkpoint/mp_rank_00_model_states.pt \
  ++module.model.lora_checkpoint=/checkpoints/12_08_2024_09_58_16/checkpoints/0-24250 \
  ++run.llm=meta-llama/Meta-Llama-3-8B-Instruct \
  ++run.vit_model=vit_so400m_patch14_siglip_384 \
  ++run.train_padding_side="left" \
  module.model.adapter_type="mlp" \
  module.model.load_4bit=False \
  module.model.use_flash_attn=True \
  module.model.use_lora=True \
  run.test_batch_size=2 run.test_num_workers=16 \
  trainer.devices=1 \ # multi-GPU is not supported
  '++logger.wandb.tags=[eval,maxm]'
```
</details>


## Citation

```
@article{centurio2025,
  author       = {Gregor Geigle and
                  Florian Schneider and
                  Carolin Holtermann and
                  Chris Biemann and
                  Radu Timofte and
                  Anne Lauscher and
                  Goran Glava\v{s}},
  title        = {Centurio: On Drivers of Multilingual Ability of Large Vision-Language Model},
  journal      = {arXiv},
  volume       = {abs/2501.05122},
  year         = {2025},
  url          = {https://arxiv.org/abs/2501.05122},
  eprinttype    = {arXiv},
  eprint       = {2501.05122},
}
```