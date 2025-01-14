import base64
import os
from pathlib import Path
from typing import Any

import srsly
from datasets import load_dataset
from PIL import Image

IMAGE_PLACEHOLDER_TOKEN = "<image_placeholder>"

ALL_EXPERIMENTS = {
    "bin-mc",
    "cvqa",
    "m3exam-standard",
    "m3exam",
    "m5b-vgr",
    "m5b-vlod",
    "marvl",
    "maxm",
    "mme",
    "mmu",  # keeping this typo for backwards compatibility
    "mtvqa",
    "smpqa",
    "xgqa",
    "xm3600",
    "xmmmu",
    "xvnli",
}


def load_image(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(str(image_path)).convert("RGB")


def image_to_b64_url(image_path: str | Path) -> str:
    with open(str(image_path), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    _, extension = os.path.splitext(str(image_path))
    mime_type = f"image/{extension[1:].lower()}"

    return f"data:{mime_type};base64,{encoded_string}"


def get_all_test_suite_ex_cfgs(
    cfg_root_p: Path,
    ex_allow: list[str] | None = None,
    ex_deny: list[str] | None = None,
) -> list[Path]:
    ex_cfgs_p = cfg_root_p / "experiment"
    if not ex_cfgs_p.exists():
        raise FileNotFoundError(f"Experiments Config root not found: {ex_cfgs_p}")
    test_suite_ex_cfgs = sorted(
        list(
            p
            for p in ex_cfgs_p.glob("mblipv2_test_suite_*.yaml")
            if "all" not in p.stem[-3:]
        )
    )
    if ex_allow is None and ex_deny is None:
        ex_allow_set = ALL_EXPERIMENTS
        ex_deny_set = set()
    elif ex_allow is not None and ex_deny is None:
        ex_allow_set = set(ex_allow)
        ex_deny_set = set()
    elif ex_allow is None and ex_deny is not None:
        ex_deny_set = set(ex_deny)
        ex_allow_set = ALL_EXPERIMENTS - ex_deny_set
    elif ex_allow is not None and ex_deny is not None:
        ex_allow_set = set(ex_allow)
        ex_deny_set = set(ex_deny)

        if ex_allow_set.intersection(ex_deny_set):
            raise ValueError(
                f"Datasets in {ex_allow_set=} and {ex_deny_set=} must be disjoint: {ex_allow_set.intersection(ex_deny_set)}"
            )

    test_suite_ex_cfgs = [
        p
        for p in test_suite_ex_cfgs
        if any(
            ds == p.stem.replace("mblipv2_test_suite_", "").replace(".yaml", "")
            for ds in ex_allow_set
        )
    ]

    test_suite_ex_cfgs = [
        p
        for p in test_suite_ex_cfgs
        if all(
            ds != p.stem.replace("mblipv2_test_suite_", "").replace(".yaml", "")
            for ds in ex_deny_set
        )
    ]

    return test_suite_ex_cfgs


def _load_dataspecs_cfg(
    test_suite_ex_cfg_p: Path,
    cfg_root_p: Path,
) -> dict[str, Any]:
    if not test_suite_ex_cfg_p.exists():
        raise FileNotFoundError(f"Experiment Config not found: {test_suite_ex_cfg_p}")
    test_suite_ex_cfg = srsly.yaml_loads(test_suite_ex_cfg_p.read_text())
    if "defaults" not in test_suite_ex_cfg:
        raise KeyError(
            f"Key 'defaults' not found in Experiment Config {test_suite_ex_cfg_p}"
        )
    for d in test_suite_ex_cfg["defaults"]:
        if isinstance(d, dict):
            dspecs_cfg_fn = d.get("/dataspecs@datamodule.test", None)
            if dspecs_cfg_fn:
                dspecs_cfg_fn = dspecs_cfg_fn[0]
                break
    dspecs_cfg_fn = cfg_root_p / f"dataspecs/{dspecs_cfg_fn}.yaml"
    if not dspecs_cfg_fn.exists():
        raise FileNotFoundError(f"Dataspecs Config not found: {dspecs_cfg_fn}")

    return srsly.yaml_loads(dspecs_cfg_fn.read_text())


def _load_dataspec_cfg(
    dataspec_cfg_name: str,
    cfg_root_p: Path,
) -> dict[str, Any]:
    dataspec_cfg_fn = cfg_root_p / f"dataspec/{dataspec_cfg_name}.yaml"
    if not dataspec_cfg_fn.exists():
        raise FileNotFoundError(f"Dataspec Config not found: {dataspec_cfg_fn}")
    return srsly.yaml_loads(dataspec_cfg_fn.read_text())


def _apply_target2str(
    sample: dict[str, Any],
    target2str: dict[str, str],
) -> dict[str, Any]:
    applied = {}
    for k, v in sample.items():
        if k in ["text_label", "label"]:
            applied[k] = target2str.get(v, v)
        else:
            applied[k] = v
    return applied


def load_test_suite_experiment_dataspecs(
    test_suite_ex_cfg_p: Path,
    cfg_root_p: Path,
    data_root: Path,
    images_root: Path,
) -> list[dict[str, Any]]:
    dataspecs_cfg = _load_dataspecs_cfg(test_suite_ex_cfg_p, cfg_root_p)
    dataspecs = []
    for d in dataspecs_cfg["defaults"]:
        if isinstance(d, dict):
            for dataspecs_name, dataspec_cfg_name in d.items():
                dataspec_cfg = _load_dataspec_cfg(dataspec_cfg_name, cfg_root_p)

                dataspec_lang_cfg = {}
                if dataspecs_name.startswith("/dataspec@"):
                    dataspecs_name = dataspecs_name.replace("/dataspec@", "")
                    dataspec_lang_cfg["dataspecs_name"] = dataspecs_name
                    dataspec_lang_cfg["dataspec_name"] = dataspec_cfg_name

                    # get the evaluation method
                    for ds_def in dataspec_cfg["defaults"]:
                        if isinstance(ds_def, dict):
                            for ds_def_key, ds_def_val in ds_def.items():
                                if ds_def_key == "evaluation":
                                    dataspec_lang_cfg["eval_method"] = ds_def_val

                    # get the image root
                    if "preprocessing" in dataspecs_cfg[dataspecs_name]:
                        # dataspecs specific image root
                        transform_cfg = dataspecs_cfg[dataspecs_name]["preprocessing"][
                            "method"
                        ]["set_transform"]["transform"]
                        dataspec_lang_cfg["image_root"] = transform_cfg[
                            "image_root"
                        ].replace("${run.image_root}", str(images_root))
                    else:
                        # dataspec specific image root
                        transform_cfg = dataspec_cfg["preprocessing"]["method"][
                            "set_transform"
                        ]["transform"]
                        dataspec_lang_cfg["image_root"] = transform_cfg[
                            "image_root"
                        ].replace("${run.image_root}", str(images_root))

                    # get the image extension
                    if "extension" in transform_cfg:
                        dataspec_lang_cfg["image_extension"] = transform_cfg[
                            "extension"
                        ]
                    else:
                        dataspec_lang_cfg["image_extension"] = ""

                    # get the prompt template
                    if "preprocessing" in dataspec_cfg:
                        try:
                            dataspec_lang_cfg["prompt_template"] = dataspec_cfg[
                                "preprocessing"
                            ]["method"]["map"]["function"]["template"]
                        except KeyError:
                            dataspec_lang_cfg["prompt_template"] = "{}"
                    else:
                        dataspec_lang_cfg["prompt_template"] = "{}"

                    # get target2str
                    if "preprocessing" in dataspec_cfg:
                        try:
                            dataspec_lang_cfg["target2str"] = dataspec_cfg[
                                "preprocessing"
                            ]["method"]["map"]["function"]["target2str"]
                        except KeyError:
                            dataspec_lang_cfg["target2str"] = dict()
                    else:
                        dataspec_lang_cfg["target2str"] = dict()

                    # load the actual dataset
                    load_ds_params = dataspecs_cfg[dataspecs_name]["dataset"]
                    load_ds_params["data_files"] = (
                        load_ds_params["data_files"]
                        .replace("${run.data_prefix}", str(data_root))
                        .replace("${run.train_data}", str(data_root))
                    )
                    ds = load_dataset(**load_ds_params)
                    # apply target2str
                    if dataspec_lang_cfg["target2str"]:
                        ds = ds.map(
                            lambda x: _apply_target2str(
                                x, dataspec_lang_cfg["target2str"]
                            )
                        )
                    dataspec_lang_cfg["dataset"] = ds

                    dataspecs.append(dataspec_lang_cfg)

    return dataspecs


def get_prompt_parts(
    prompt: str, image_token: str = IMAGE_PLACEHOLDER_TOKEN
) -> list[str]:
    prompt_parts = prompt.split(image_token)
    if len(prompt_parts) == 1:
        prompt_parts = [""] + prompt_parts

    return prompt_parts
