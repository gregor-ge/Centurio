from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from src.baseline_eval.data_utils import (
    ALL_EXPERIMENTS,
    _load_dataspecs_cfg,
    get_all_test_suite_ex_cfgs,
)
from src.baseline_eval.models import get_all_supported_baseline_models

LANGS_DF = pd.read_csv("flores2iso2tax.tsv", sep="\t")


def get_model_and_dataspec(eval_dir_name: Path | str) -> dict[str, str]:
    eval_dir_name = Path(eval_dir_name).name

    model_id, dataspec = eval_dir_name.rsplit("__", 1)
    model_id = model_id.replace("__", "/")

    return {"model_id": model_id, "dataspec": dataspec}


def get_dataspecs_names_from_dataspec(dataspec: str, cfg_root_p: Path) -> list[str]:
    ex_cfgs = get_all_test_suite_ex_cfgs(cfg_root_p=cfg_root_p)

    dataspec_name_to_ex_cfg_name_mapping = {
        "m3exam_cycle": "m3exam",
        "m3exam": "m3exam-standard",
        "babelimagenet-mc": "bin-mc",
        "m5b_vlod": "m5b-vlod",
        "m5b_vgr": "m5b-vgr",
        "mmmu": "mmu",
    }
    dataspec = dataspec_name_to_ex_cfg_name_mapping.get(dataspec, dataspec)

    ex_cfg = list(
        filter(
            lambda ex_cfg: ex_cfg.name == f"mblipv2_test_suite_{dataspec}.yaml", ex_cfgs
        )
    )
    if len(ex_cfg) == 0:
        raise ValueError(f"Dataspec {dataspec} not found in {cfg_root_p}")
    elif len(ex_cfg) > 1:
        raise ValueError(f"Multiple dataspecs {dataspec} found in {cfg_root_p}")
    ex_cfg = ex_cfg[0]
    dataspecs_cfg = _load_dataspecs_cfg(
        test_suite_ex_cfg_p=ex_cfg, cfg_root_p=cfg_root_p
    )
    dataspecs = []
    for default in dataspecs_cfg["defaults"]:
        if isinstance(default, dict):
            for dataspecs_name, dataspec_name in default.items():
                if dataspecs_name.startswith("/dataspec@"):
                    dataspecs.append(dataspecs_name.split("/dataspec@")[1])

    return dataspecs


def get_finished_dataspecs(resp: Path):
    finished_dataspecs = list(
        p.name.replace("_eval_result.json", "") for p in resp.glob("*_eval_result.json")
    )
    return finished_dataspecs


def get_inprogress_dataspecs(resp: Path):
    inprog_dataspecs = set(p.name.replace(".jsonl", "") for p in resp.glob("*.jsonl"))
    finished_dataspecs = set(get_finished_dataspecs(resp))
    inprog_dataspecs = list(inprog_dataspecs - finished_dataspecs)

    return inprog_dataspecs


def get_all_dataspec_names(cfg_root_p: Path) -> list[str]:
    ex_cfgs = get_all_test_suite_ex_cfgs(cfg_root_p=cfg_root_p)
    # ex_cfg to dataspec mapping
    ex_cfg_name_to_dataspec_name_mapping = {
        "m3exam": "m3exam_cycle",
        "m3exam-standard": "m3exam",
        "bin-mc": "babelimagenet-mc",
        "m5b-vlod": "m5b_vlod",
        "m5b-vgr": "m5b_vgr",
        "mmu": "mmmu",
    }

    all_dataspec_names = list(
        p.name.replace("mblipv2_test_suite_", "").replace(".yaml", "") for p in ex_cfgs
    )
    all_dataspec_names = list(
        ex_cfg_name_to_dataspec_name_mapping.get(dataspec, dataspec)
        for dataspec in all_dataspec_names
    )
    return all_dataspec_names


def get_dataspec_result_status(
    results_root: Path,
    cfg_root_p: Path,
) -> pd.DataFrame:
    results = {
        "model_id": [],
        "dataspec": [],
        "done": [],
        "inprogress": [],
        "todo": [],
        "completed": [],
    }

    # get all from the results
    for res_p in results_root.iterdir():
        if res_p.is_dir():
            res_metadata = get_model_and_dataspec(res_p)
            results["model_id"].append(res_metadata["model_id"])
            results["dataspec"].append(res_metadata["dataspec"])

            all_dataspecs_names = get_dataspecs_names_from_dataspec(
                res_metadata["dataspec"], cfg_root_p=cfg_root_p
            )

            finished_dataspecs = get_finished_dataspecs(res_p)
            results["done"].append(finished_dataspecs)

            todo = list(set(all_dataspecs_names) - set(finished_dataspecs))
            results["todo"].append(todo)

            inprogress_dataspecs = get_inprogress_dataspecs(res_p)
            results["inprogress"].append(inprogress_dataspecs)

            results["completed"].append(
                len(finished_dataspecs) / len(all_dataspecs_names)
            )

    df = pd.DataFrame(results)

    return df.sort_values(by=["model_id", "dataspec"]).reset_index(drop=True)


def get_model_result_status(
    dataspec_result_status_df: pd.DataFrame, cfg_root_p: Path
) -> pd.DataFrame:
    model_status = {
        "model_id": [],
        "done": [],
        "inprogress": [],
        "todo": [],
        "completed": [],
    }
    all_dataspec_names = get_all_dataspec_names(cfg_root_p)

    model_ids = dataspec_result_status_df["model_id"].unique()
    for model_id in model_ids:
        model_status["model_id"].append(model_id)

        completed_dataspec = dataspec_result_status_df[
            (dataspec_result_status_df["model_id"] == model_id)
            & (dataspec_result_status_df["completed"] == 1.0)
        ]

        model_status["done"].append(completed_dataspec["dataspec"].tolist())

        inprogress_dataspec = dataspec_result_status_df[
            (dataspec_result_status_df["model_id"] == model_id)
            & (dataspec_result_status_df["completed"] < 1.0)
        ]
        model_status["inprogress"].append(inprogress_dataspec["dataspec"].tolist())

        todo_dataspec = list(
            set(all_dataspec_names)
            - set(completed_dataspec["dataspec"].tolist())
            - set(inprogress_dataspec["dataspec"].tolist())
        )
        model_status["todo"].append(todo_dataspec)

        model_status["completed"].append(
            len(completed_dataspec) / len(all_dataspec_names)
        )

    return pd.DataFrame(model_status)


def generate_run_script_commands(
    model_result_status_df: pd.DataFrame | None = None,
    dataset_filter: Callable[[str], bool] | None = None,
    model_filter: Callable[[str], bool] | None = None,
    model_id: str | None = None,
    datasets: list[str] | None = None,
    yes: bool = False,
    only_eval: bool = False,
    inprogress: bool = False,
    done: bool = False,
    add_answer_in_one_word_prompt: bool = False,
    mamba_env_name: str = "mblipv2",
):
    if model_result_status_df is not None:
        for _, row in model_result_status_df.iterrows():
            model_id = row["model_id"]
            if model_filter is None or model_filter(model_id):
                print(f"# {model_id}")
                if "parrot" in model_id.lower():
                    mamba_env_name = "parrot"
                elif "palo-" in model_id.lower():
                    mamba_env_name = "palo"
                else:
                    mamba_env_name = "mblipv2"
                cols = [row["todo"]]
                if done:
                    cols.append(row["done"])
                if inprogress:
                    cols.append(row["inprogress"])
                for ds in set(sum(cols, [])):
                    if dataset_filter is None or dataset_filter(ds):
                        # custom mapping
                        if ds.startswith("m5b"):
                            ds = ds.replace("_", "-")
                        elif ds == "mmmu":
                            ds = "mmu"
                        elif ds == "babelimagenet-mc":
                            ds = "bin-mc"
                        elif ds == "m3exam":
                            ds = "m3exam-standard"
                        elif ds == "m3exam_cycle":
                            ds = "m3exam"
                        cmd = (
                            "CUDA_VISIBLE_DEVICES=0 "
                            f"MAMBA_ENV_NAME='{mamba_env_name}' "
                            "VLLM_CONFIGURE_LOGGING=0 "
                            f"HF_MODEL_ID='{model_id}' "
                            f"{'YES=True' if yes else ''} "
                            f"{'ONLY_EVAL=True' if only_eval else ''} "
                            f"{'ADD_ANSWER_IN_ONE_WORD_PROMPT=True' if add_answer_in_one_word_prompt and ('pangea' in model_id.lower() or 'palo' in model_id.lower())  else ''} "
                            f"scripts/run_baseline_eval.sh '{ds}'"
                        )
                        print(cmd)
                print()

    elif model_id is not None:
        all_supported_baseline_models = get_all_supported_baseline_models()
        if model_id not in all_supported_baseline_models:
            raise ValueError(
                f"Model {model_id} not found in supported models: {all_supported_baseline_models}"
            )
        if datasets is None:
            datasets = list(ALL_EXPERIMENTS)

        if "parrot" in model_id.lower():
            mamba_env_name = "parrot"
        elif "palo-" in model_id.lower():
            mamba_env_name = "palo"
        for ds in datasets:
            cmd = (
                "CUDA_VISIBLE_DEVICES=0 "
                f"MAMBA_ENV_NAME='{mamba_env_name}' "
                "VLLM_CONFIGURE_LOGGING=0 "
                f"HF_MODEL_ID='{model_id}' "
                f"{'YES=True' if yes else ''} "
                f"{'ONLY_EVAL=True' if only_eval else ''} "
                f"{'ADD_ANSWER_IN_ONE_WORD_PROMPT=True' if add_answer_in_one_word_prompt and ('pangea' in model_id.lower() or 'palo' in model_id.lower())  else ''} "
                f"scripts/run_baseline_eval.sh '{ds}'"
            )
            print(cmd)


def get_all_scores_for_dataspec(
    dataspec: str,
    results_root: Path,
) -> pd.DataFrame:
    res_fns = list(
        p
        for p in results_root.rglob("*results.json*")
        if p.parent.name.split("__")[-1] == dataspec
    )
    if len(res_fns) == 0:
        raise ValueError(f"No results found for dataspec {dataspec}")
    res = (
        pd.concat([pd.read_json(p) for p in res_fns])
        .sort_values("model_id")
        .reset_index(drop=True)
    )

    def get_lang(dataspecs_name: str) -> str:
        if dataspecs_name.startswith("m5b"):
            lang = dataspecs_name.split("_")[2]
        else:
            lang = dataspecs_name.split("_")[1]
        return get_lang_code(lang)

    res["lang"] = res["dataspecs_name"].apply(get_lang)
    return res


def get_lang_code(
    language: str,
) -> str:
    code = "ISO-639-1"
    if language == "chinese":
        return "zh"
    if language in LANGS_DF[code].values:
        return language
    if language in LANGS_DF["iso639_name"].values:
        return LANGS_DF[LANGS_DF["iso639_name"] == language][code].values[0]
    elif language in LANGS_DF["flores200_name"].values:
        return LANGS_DF[LANGS_DF["flores200_name"] == language][code].values[0]
    elif language in LANGS_DF["tax_name"].values:
        return LANGS_DF[LANGS_DF["tax_name"] == language][code].values[0]
    else:
        return language


def build_scores_table(
    model_id: str,
    results_root: Path,
    verbose: bool = False,
) -> pd.DataFrame:
    all_scores = {
        "model_id": [],
        "maxm/en/vqa_acc": [],
        "maxm/fr/vqa_acc": [],
        "maxm/hi/vqa_acc": [],
        "maxm/he/iw/vqa_acc": [],
        "maxm/ro/vqa_acc": [],
        "maxm/th/vqa_acc": [],
        "maxm/zh/vqa_acc": [],
        # "maxm/average/vqa_acc": [],
        # "maxm/en_average/vqa_acc": [],
        "xgqa/bn/acc": [],
        "xgqa/bn/relaxed_acc": [],
        "xgqa/de/acc": [],
        "xgqa/de/relaxed_acc": [],
        "xgqa/en/acc": [],
        "xgqa/en/relaxed_acc": [],
        "xgqa/id/acc": [],
        "xgqa/id/relaxed_acc": [],
        "xgqa/ko/acc": [],
        "xgqa/ko/relaxed_acc": [],
        "xgqa/pt/acc": [],
        "xgqa/pt/relaxed_acc": [],
        "xgqa/ru/acc": [],
        "xgqa/ru/relaxed_acc": [],
        "xgqa/zh/acc": [],
        "xgqa/zh/relaxed_acc": [],
        # "xgqa/average/acc": [],
        # "xgqa/average/relaxed_acc": [],
        # "xgqa/en_average/acc": [],
        # "xgqa/en_average/relaxed_acc": [],
        "xm3600/ar/CIDEr": [],
        "xm3600/bn/CIDEr": [],
        "xm3600/cs/CIDEr": [],
        "xm3600/da/CIDEr": [],
        "xm3600/de/CIDEr": [],
        "xm3600/el/CIDEr": [],
        "xm3600/en/CIDEr": [],
        "xm3600/es/CIDEr": [],
        "xm3600/fa/CIDEr": [],
        "xm3600/fi/CIDEr": [],
        "xm3600/fil/CIDEr": [],
        "xm3600/fr/CIDEr": [],
        "xm3600/he/iw/CIDEr": [],
        "xm3600/hi/CIDEr": [],
        "xm3600/hr/CIDEr": [],
        "xm3600/hu/CIDEr": [],
        "xm3600/id/CIDEr": [],
        "xm3600/it/CIDEr": [],
        "xm3600/ja/CIDEr": [],
        "xm3600/ko/CIDEr": [],
        "xm3600/mi/CIDEr": [],
        "xm3600/nl/CIDEr": [],
        "xm3600/no/CIDEr": [],
        "xm3600/pl/CIDEr": [],
        "xm3600/pt/CIDEr": [],
        "xm3600/quz/CIDEr": [],
        "xm3600/ro/CIDEr": [],
        "xm3600/ru/CIDEr": [],
        "xm3600/sv/CIDEr": [],
        "xm3600/sw/CIDEr": [],
        "xm3600/te/CIDEr": [],
        "xm3600/th/CIDEr": [],
        "xm3600/tr/CIDEr": [],
        "xm3600/uk/CIDEr": [],
        "xm3600/vi/CIDEr": [],
        "xm3600/zh/CIDEr": [],
        # "xm3600/average/CIDEr": [],
        # "xm3600/en_average/CIDEr": [],
        "bin-mc/af/relaxed_acc": [],
        "bin-mc/am/relaxed_acc": [],
        "bin-mc/cs/relaxed_acc": [],
        "bin-mc/el/relaxed_acc": [],
        "bin-mc/en/relaxed_acc": [],
        "bin-mc/es/relaxed_acc": [],
        "bin-mc/fa/relaxed_acc": [],
        "bin-mc/fi/relaxed_acc": [],
        "bin-mc/ha/relaxed_acc": [],
        "bin-mc/hr/relaxed_acc": [],
        "bin-mc/hu/relaxed_acc": [],
        "bin-mc/ja/relaxed_acc": [],
        "bin-mc/mi/relaxed_acc": [],
        "bin-mc/nl/relaxed_acc": [],
        "bin-mc/no/relaxed_acc": [],
        "bin-mc/pl/relaxed_acc": [],
        "bin-mc/ro/relaxed_acc": [],
        "bin-mc/ta/relaxed_acc": [],
        "bin-mc/te/relaxed_acc": [],
        "bin-mc/zu/relaxed_acc": [],
        # "bin-mc/average/relaxed_acc": [],
        # "bin-mc/en_average/relaxed_acc": [],
        "smpqa/ar/ground - acc": [],
        "smpqa/ar/ground - relaxed_acc": [],
        "smpqa/ar/name - acc": [],
        "smpqa/ar/name - relaxed_acc": [],
        "smpqa/de/ground - acc": [],
        "smpqa/de/ground - relaxed_acc": [],
        "smpqa/de/name - acc": [],
        "smpqa/de/name - relaxed_acc": [],
        "smpqa/en/ground - acc": [],
        "smpqa/en/ground - relaxed_acc": [],
        "smpqa/en/name - acc": [],
        "smpqa/en/name - relaxed_acc": [],
        "smpqa/hi/ground - acc": [],
        "smpqa/hi/ground - relaxed_acc": [],
        "smpqa/hi/name - acc": [],
        "smpqa/hi/name - relaxed_acc": [],
        "smpqa/id/ground - acc": [],
        "smpqa/id/ground - relaxed_acc": [],
        "smpqa/id/name - acc": [],
        "smpqa/id/name - relaxed_acc": [],
        "smpqa/it/ground - acc": [],
        "smpqa/it/ground - relaxed_acc": [],
        "smpqa/it/name - acc": [],
        "smpqa/it/name - relaxed_acc": [],
        "smpqa/ko/ground - acc": [],
        "smpqa/ko/ground - relaxed_acc": [],
        "smpqa/ko/name - acc": [],
        "smpqa/ko/name - relaxed_acc": [],
        "smpqa/ru/ground - acc": [],
        "smpqa/ru/ground - relaxed_acc": [],
        "smpqa/ru/name - acc": [],
        "smpqa/ru/name - relaxed_acc": [],
        "smpqa/th/ground - acc": [],
        "smpqa/th/ground - relaxed_acc": [],
        "smpqa/th/name - acc": [],
        "smpqa/th/name - relaxed_acc": [],
        "smpqa/zh/ground - acc": [],
        "smpqa/zh/ground - relaxed_acc": [],
        "smpqa/zh/name - acc": [],
        "smpqa/zh/name - relaxed_acc": [],
        "smpqa/zu/ground - acc": [],
        "smpqa/zu/ground - relaxed_acc": [],
        "smpqa/zu/name - acc": [],
        "smpqa/zu/name - relaxed_acc": [],
        # "smpqa/average/ground - acc": [],
        # "smpqa/average/ground - relaxed_acc": [],
        # "smpqa/en_average/ground - acc": [],
        # "smpqa/en_average/ground - relaxed_acc": [],
        # "smpqa/average/name - acc": [],
        # "smpqa/average/name - relaxed_acc": [],
        # "smpqa/en_average/name - acc": [],
        # "smpqa/en_average/name - relaxed_acc": [],
        "xvnli/ar/acc": [],
        "xvnli/ar/relaxed_acc": [],
        "xvnli/en/acc": [],
        "xvnli/en/relaxed_acc": [],
        "xvnli/es/acc": [],
        "xvnli/es/relaxed_acc": [],
        "xvnli/fr/acc": [],
        "xvnli/fr/relaxed_acc": [],
        "xvnli/ru/acc": [],
        "xvnli/ru/relaxed_acc": [],
        # "xvnli/average/acc": [],
        # "xvnli/average/relaxed_acc": [],
        # "xvnli/en_average/acc": [],
        # "xvnli/en_average/relaxed_acc": [],
        "mme/en/OCR - acc": [],
        "mme/en/OCR - acc_plus": [],
        "mme/en/OCR - score": [],
        "mme/en/artwork - acc": [],
        "mme/en/artwork - acc_plus": [],
        "mme/en/artwork - score": [],
        "mme/en/celebrity - acc": [],
        "mme/en/celebrity - acc_plus": [],
        "mme/en/celebrity - score": [],
        "mme/en/code_reasoning - acc": [],
        "mme/en/code_reasoning - acc_plus": [],
        "mme/en/code_reasoning - score": [],
        "mme/en/color - acc": [],
        "mme/en/color - acc_plus": [],
        "mme/en/color - score": [],
        "mme/en/commonsense_reasoning - acc": [],
        "mme/en/commonsense_reasoning - acc_plus": [],
        "mme/en/commonsense_reasoning - score": [],
        "mme/en/count - acc": [],
        "mme/en/count - acc_plus": [],
        "mme/en/count - score": [],
        "mme/en/existence - acc": [],
        "mme/en/existence - acc_plus": [],
        "mme/en/existence - score": [],
        "mme/en/landmark - acc": [],
        "mme/en/landmark - acc_plus": [],
        "mme/en/landmark - score": [],
        "mme/en/numerical_calculation - acc": [],
        "mme/en/numerical_calculation - acc_plus": [],
        "mme/en/numerical_calculation - score": [],
        "mme/en/position - acc": [],
        "mme/en/position - acc_plus": [],
        "mme/en/position - score": [],
        "mme/en/posters - acc": [],
        "mme/en/posters - acc_plus": [],
        "mme/en/posters - score": [],
        "mme/en/scene - acc": [],
        "mme/en/scene - acc_plus": [],
        "mme/en/scene - score": [],
        "mme/en/text_translation - acc": [],
        "mme/en/text_translation - acc_plus": [],
        "mme/en/text_translation - score": [],
        "mme/en/test - total_cognition": [],
        "mme/en/test - total_perception": [],
        "mmu/en/relaxed_acc": [],
        "mtvqa/ar/acc": [],
        "mtvqa/ar/relaxed_acc": [],
        "mtvqa/de/acc": [],
        "mtvqa/de/relaxed_acc": [],
        "mtvqa/fr/acc": [],
        "mtvqa/fr/relaxed_acc": [],
        "mtvqa/it/acc": [],
        "mtvqa/it/relaxed_acc": [],
        "mtvqa/ja/acc": [],
        "mtvqa/ja/relaxed_acc": [],
        "mtvqa/ko/acc": [],
        "mtvqa/ko/relaxed_acc": [],
        "mtvqa/ru/acc": [],
        "mtvqa/ru/relaxed_acc": [],
        "mtvqa/th/acc": [],
        "mtvqa/th/relaxed_acc": [],
        "mtvqa/vi/acc": [],
        "mtvqa/vi/relaxed_acc": [],
        # "mtvqa/average/acc": [],
        # "mtvqa/average/relaxed_acc": [],
        # "mtvqa/en_average/acc": [],
        # "mtvqa/en_average/relaxed_acc": [],
        "m5b-vgr/am/acc": [],
        "m5b-vgr/am/relaxed_acc": [],
        "m5b-vgr/ber/acc": [],
        "m5b-vgr/ber/relaxed_acc": [],
        "m5b-vgr/bn/acc": [],
        "m5b-vgr/bn/relaxed_acc": [],
        "m5b-vgr/de/acc": [],
        "m5b-vgr/de/relaxed_acc": [],
        "m5b-vgr/en/acc": [],
        "m5b-vgr/en/relaxed_acc": [],
        "m5b-vgr/fil/acc": [],
        "m5b-vgr/fil/relaxed_acc": [],
        "m5b-vgr/ha/acc": [],
        "m5b-vgr/ha/relaxed_acc": [],
        "m5b-vgr/hi/acc": [],
        "m5b-vgr/hi/relaxed_acc": [],
        "m5b-vgr/ru/acc": [],
        "m5b-vgr/ru/relaxed_acc": [],
        "m5b-vgr/sw/acc": [],
        "m5b-vgr/sw/relaxed_acc": [],
        "m5b-vgr/th/acc": [],
        "m5b-vgr/th/relaxed_acc": [],
        "m5b-vgr/zu/acc": [],
        "m5b-vgr/zu/relaxed_acc": [],
        # "m5b-vgr/average/acc": [],
        # "m5b-vgr/average/relaxed_acc": [],
        # "m5b-vgr/en_average/acc": [],
        # "m5b-vgr/en_average/relaxed_acc": [],
        "m5b-vlod/am/acc": [],
        "m5b-vlod/am/relaxed_acc": [],
        "m5b-vlod/ber/acc": [],
        "m5b-vlod/ber/relaxed_acc": [],
        "m5b-vlod/bn/acc": [],
        "m5b-vlod/bn/relaxed_acc": [],
        "m5b-vlod/de/acc": [],
        "m5b-vlod/de/relaxed_acc": [],
        "m5b-vlod/en/acc": [],
        "m5b-vlod/en/relaxed_acc": [],
        "m5b-vlod/fil/acc": [],
        "m5b-vlod/fil/relaxed_acc": [],
        "m5b-vlod/ha/acc": [],
        "m5b-vlod/ha/relaxed_acc": [],
        "m5b-vlod/hi/acc": [],
        "m5b-vlod/hi/relaxed_acc": [],
        "m5b-vlod/ru/acc": [],
        "m5b-vlod/ru/relaxed_acc": [],
        "m5b-vlod/sw/acc": [],
        "m5b-vlod/sw/relaxed_acc": [],
        "m5b-vlod/th/acc": [],
        "m5b-vlod/th/relaxed_acc": [],
        "m5b-vlod/zu/acc": [],
        "m5b-vlod/zu/relaxed_acc": [],
        # "m5b-vlod/average/acc": [],
        # "m5b-vlod/average/relaxed_acc": [],
        # "m5b-vlod/en_average/acc": [],
        # "m5b-vlod/en_average/relaxed_acc": [],
        "marvl/id/acc": [],
        "marvl/id/relaxed_acc": [],
        "marvl/sw/acc": [],
        "marvl/sw/relaxed_acc": [],
        "marvl/ta/acc": [],
        "marvl/ta/relaxed_acc": [],
        "marvl/tr/acc": [],
        "marvl/tr/relaxed_acc": [],
        "marvl/zh/acc": [],
        "marvl/zh/relaxed_acc": [],
        "marvl/en/acc": [],
        "marvl/en/relaxed_acc": [],
        # "marvl/average/acc": [],
        # "marvl/average/relaxed_acc": [],
        # "marvl/en_average/acc": [],
        # "marvl/en_average/relaxed_acc": [],
        "xmmmu/ar/relaxed_acc": [],
        "xmmmu/en/relaxed_acc": [],
        "xmmmu/fr/relaxed_acc": [],
        "xmmmu/hi/relaxed_acc": [],
        "xmmmu/id/relaxed_acc": [],
        "xmmmu/ja/relaxed_acc": [],
        "xmmmu/pt/relaxed_acc": [],
        # "xmmmu/average/relaxed_acc": [],
        # "xmmmu/en_average/relaxed_acc": [],
        "m3exam-standard/af/relaxed_acc": [],
        "m3exam-standard/zh/relaxed_acc": [],
        "m3exam-standard/en/relaxed_acc": [],
        "m3exam-standard/it/relaxed_acc": [],
        "m3exam-standard/pt/relaxed_acc": [],
        "m3exam-standard/th/relaxed_acc": [],
        "m3exam-standard/vi/relaxed_acc": [],
        # "m3exam-standard/average/relaxed_acc": [],
        # "m3exam-standard/en_average/relaxed_acc": [],
    }
    all_scores["model_id"].append(model_id)
    scores_dfs = {}
    for k in tqdm(all_scores.keys()):
        if k != "model_id":
            ds = k.split("/")[0]
            lang = k.split("/")[1]
            score = k.split("/")[-1]

            if "average" in lang:
                if verbose:
                    print(f"Skipping {k}")
                all_scores[k].append("N/A")
                continue

            # custom mapping
            ds_name = ds
            if ds == "m3exam-standard":
                ds_name = "m3exam"
            if ds == "bin-mc":
                ds_name = "babelimagenet-mc"
            if ds == "mmu":
                ds_name = "mmmu"
            if ds.startswith("m5b"):
                ds_name = ds.replace("-", "_")

            scores_name = score
            if ds == "mme":
                scores_name = score.replace(" - ", "/")
                if "test/" in scores_name:
                    scores_name = scores_name.split("/")[1]
            if ds == "maxm":
                scores_name = "acc"

            lang_name = lang
            if ds == "maxm" and lang == "he":
                lang_name = "iw"
            if ds == "mtvqa" and lang == "ko":
                lang_name = "kr"

            # load the scores for the dataspec
            if ds not in scores_dfs:
                scores_df = get_all_scores_for_dataspec(ds_name, results_root)
                scores_dfs[ds] = scores_df
            else:
                scores_df = scores_dfs[ds]

            if model_id not in scores_df["model_id"].values:
                if verbose:
                    print(
                        f"Missing {model_id} score for {ds_name} {lang_name} {scores_name}"
                    )
                all_scores[k].append("N/A")
                continue

            # get the score that matches the key
            scores_rows = scores_df[
                (scores_df["model_id"] == model_id) & (scores_df["lang"] == lang_name)
            ]

            if ds == "smpqa":
                if "ground" in scores_name:
                    scores_row = scores_rows[
                        scores_rows["dataspecs_name"].str.contains("ground")
                    ]
                elif "name" in scores_name:
                    scores_row = scores_rows[
                        scores_rows["dataspecs_name"].str.contains("name")
                    ]
                else:
                    if verbose:
                        display(scores_rows)
                        print(ds_name, lang_name, scores_name)
                        raise ValueError(
                            f"Could not find score for {model_id} {ds_name} {lang_name} {scores_name}"
                        )

                if len(scores_row) == 0:
                    if verbose:
                        display(scores_rows)
                        print(ds_name, lang_name, scores_name)
                        raise ValueError(
                            f"Could not find score for {model_id} {ds_name} {lang_name} {scores_name}"
                        )
                elif len(scores_row) > 1:
                    if verbose:
                        display(scores_rows)
                        print(ds_name, lang_name, scores_name)
                        raise ValueError(
                            f"Multiple scores found for {model_id} {ds_name} {lang_name} {scores_name}"
                        )
                if "relaxed_acc" in scores_name:
                    score_val = scores_row["relaxed_acc"].values[0]
                elif "acc" in scores_name:
                    score_val = scores_row["acc"].values[0]
                else:
                    if verbose:
                        display(scores_rows)
                        print(ds_name, lang_name, scores_name)
                        raise ValueError(
                            f"Could not find score for {model_id} {ds_name} {lang_name} {scores_name}"
                        )
                all_scores[k].append(score_val)

            else:
                try:
                    score_val = scores_rows[scores_name].values[0]
                    all_scores[k].append(score_val)
                except:
                    if verbose:
                        display(scores_rows)
                        print(ds_name, lang_name, scores_name)
                        raise ValueError(
                            f"Could not find score for {model_id} {ds_name} {lang_name} {scores_name}"
                        )
                    all_scores[k].append("N/A")

    return pd.DataFrame(all_scores)


def build_big_scores_table(
    results_root: Path, config_root: Path, verbose: bool = False
) -> pd.DataFrame:
    models = get_model_result_status(
        get_dataspec_result_status(results_root, config_root), config_root
    ).model_id.unique()
    all_scores = []
    for model_id in tqdm(models):
        model_scores = build_scores_table(model_id, results_root, verbose=verbose)
        all_scores.append(model_scores)
    all_scores_df = pd.concat(all_scores).reset_index(drop=True)
    return all_scores_df
