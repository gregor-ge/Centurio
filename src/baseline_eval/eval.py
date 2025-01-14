import importlib
from pathlib import Path
from typing import Callable

import pandas as pd
import srsly
from loguru import logger

EVAL_METHOD_DEFAULT_KWARGS = {
    "src.tasks.vllm.evaluation.classification_evaluation": {
        "vqa_process": False,
        "print_examples": 10,
    },
    "src.tasks.vllm.evaluation.caption_evaluation": {
        "print_examples": 10,
    },
    "src.tasks.vllm.evaluation.mme_eval": {
        "print_examples": 10,
    },
    "src.tasks.vllm.evaluation.vqa_maxm_classification_evaluation": {
        "print_examples": 10,
    },
    "src.tasks.vllm.evaluation.mc_cycle_eval": {
        "print_examples": 10,
    },
    "src.tasks.vllm.evaluation.just_dump_evaluation": {
        "print_examples": 10,
    },
}

EVAL_METHOD_DATASPEC_KWARGS = {
    "src.tasks.vllm.evaluation.classification_evaluation": {
        "xvnli": {},
        "xmmmu": {},
        "xgqa": {},
        "smpqa": {},
        "mtvqa": {},
        "mmmu": {},
        "marvl": {},
        "m5b_vlod": {},
        "m5b_vgr": {},
        "m3exam": {},
        "babelimagenet-mc": {},
    },
    "src.tasks.vllm.evaluation.mc_cycle_eval": {
        "m3exam_cycle": {},
    },
    "src.tasks.vllm.evaluation.just_dump_evaluation": {
        "cvqa": {},
    },
    "src.tasks.vllm.evaluation.caption_evaluation": {
        "xm3600": {},
    },
    "src.tasks.vllm.evaluation.mme_eval": {
        "mme": {},
    },
    "src.tasks.vllm.evaluation.vqa_maxm_classification_evaluation": {
        "maxm": {
            "vqa_process": False,
        }
    },
}

DS_EVAL_METHOD = {
    "xvnli": "src.tasks.vllm.evaluation.classification_evaluation",
    "xmmmu": "src.tasks.vllm.evaluation.classification_evaluation",
    "xgqa": "src.tasks.vllm.evaluation.classification_evaluation",
    "smpqa": "src.tasks.vllm.evaluation.classification_evaluation",
    "mtvqa": "src.tasks.vllm.evaluation.classification_evaluation",
    "mmmu": "src.tasks.vllm.evaluation.classification_evaluation",
    "xm3600": "src.tasks.vllm.evaluation.caption_evaluation",
    "mme": "src.tasks.vllm.evaluation.mme_eval",
    "maxm": "src.tasks.vllm.evaluation.vqa_maxm_classification_evaluation",
    "marvl": "src.tasks.vllm.evaluation.classification_evaluation",
    "m5b_vlod": "src.tasks.vllm.evaluation.classification_evaluation",
    "m5b_vgr": "src.tasks.vllm.evaluation.classification_evaluation",
    "m3exam": "src.tasks.vllm.evaluation.classification_evaluation",
    "m3exam_cycle": "src.tasks.vllm.evaluation.mc_cycle_eval",
    "cvqa": "src.tasks.vllm.evaluation.just_dump_evaluation",
    "babelimagenet-mc": "src.tasks.vllm.evaluation.classification_evaluation",
}


def _load_eval_method(dataspec_name: str) -> tuple[str, Callable]:
    eval_method_path = DS_EVAL_METHOD[dataspec_name]

    module_name, method_name = eval_method_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    method = getattr(module, method_name)

    return eval_method_path, method


def run_eval(
    dataspec_name: str,
    dataspecs_name: str,
    sample_outputs: pd.DataFrame,
    output_dataspec_root: Path,
    data_root: Path,
) -> dict:
    # mandatory kwargs: image_ids, text_labels, captions
    eval_method_path, eval_method = _load_eval_method(dataspec_name)

    captions = sample_outputs["output_text"].tolist()
    image_ids = sample_outputs["image_id"].tolist()
    text_labels = sample_outputs["text_label"].tolist()

    if isinstance(captions[0], list):
        if not isinstance(image_ids[0], list):
            image_ids = [[ii] for ii in sample_outputs["image_id"].tolist()]
        if not isinstance(text_labels[0], list):
            text_labels = [[tl] for tl in sample_outputs["text_label"].tolist()]

    if not len(captions) == len(image_ids) == len(text_labels):
        raise ValueError(
            f"Length of captions (output_texts) ({len(captions)}), image_ids ({len(image_ids)}) and text_labels ({len(text_labels)}) must be equal"
        )

    image_ids_type = type(image_ids[0])
    text_labels_type = type(text_labels[0])
    captions_type = type(captions[0])
    if image_ids_type != text_labels_type or text_labels_type != captions_type:
        raise ValueError(
            f"Types of image_ids ({image_ids_type}), text_labels ({text_labels_type}) and captions ({captions_type}) must be the same"
        )

    default_kwargs = EVAL_METHOD_DEFAULT_KWARGS[eval_method_path]
    dataspec_kwargs = EVAL_METHOD_DATASPEC_KWARGS[eval_method_path][dataspec_name]

    all_kwargs = {
        "image_ids": image_ids,
        "text_labels": text_labels,
        "captions": captions,
        **dataspec_kwargs,
        **default_kwargs,
    }

    if eval_method_path == "src.tasks.vllm.evaluation.caption_evaluation":
        if dataspec_name == "xm3600":
            xm3600_lang = dataspecs_name.split("_")[-1]
            anno_file = data_root / f"xm3600/xm3600_coco_{xm3600_lang}.json"
            if not anno_file.exists():
                raise FileNotFoundError(f"Annotation file {anno_file} not found")
            all_kwargs["annotation_file"] = str(anno_file)
    elif (
        "src.tasks.vllm.evaluation.vqa_maxm_classification_evaluation"
        in eval_method_path
    ):
        # for maxm, we need to pass the text_labels as a list of lists
        text_labels = [[tl] for tl in sample_outputs["text_label"].tolist()]
        all_kwargs["text_labels"] = text_labels

    logger.info(
        f"Running eval for {dataspecs_name} using {eval_method_path} with ds_kwargs: {dataspec_kwargs} and default_kwargs: {default_kwargs}"
    )
    eval_result = eval_method(**all_kwargs)

    # hack for maxm to transform the float eval result to a dict
    if "vqa_maxm_classification_evaluation" in eval_method_path:
        eval_result = {"acc": eval_result}

    logger.info(
        f"Eval results for {dataspecs_name}: {srsly.json_dumps(eval_result, indent=2)}"
    )

    output_eval_path = output_dataspec_root / f"{dataspecs_name}_eval_result.json"
    output_eval_path.write_text(srsly.json_dumps(eval_result, indent=2))
    logger.info(f"Eval results for {dataspecs_name} saved to {output_eval_path}")

    return eval_result
