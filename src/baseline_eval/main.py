from pathlib import Path
from pprint import pformat

import pandas as pd
from fire import Fire
from lightning import seed_everything
from loguru import logger
from tqdm.auto import tqdm

# TODO logger: see https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
# """
# class InterceptHandler(logging.Handler):
#     def emit(self, record: logging.LogRecord) -> None:
#         # Get corresponding Loguru level if it exists.
#         level: str | int
#         try:
#             level = logger.level(record.levelname).name
#         except ValueError:
#             level = record.levelno

#         # Find caller from where originated the logged message.
#         frame, depth = inspect.currentframe(), 0
#         while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
#             frame = frame.f_back
#             depth += 1

#         logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
# """

from src.baseline_eval.data_utils import (
    get_all_test_suite_ex_cfgs,
    load_test_suite_experiment_dataspecs,
)
from src.baseline_eval.eval import run_eval
from src.baseline_eval.models import load_model, BaselineModel


def _get_image_paths(sample: dict, image_root: Path, image_extension: str) -> list[str]:
    image_paths = []
    if isinstance(sample["image_id"], list):
        for image_id in sample["image_id"]:
            image_fn = image_id + image_extension
            image_path = Path(image_root) / image_fn
            image_paths.append(image_path)
    else:
        image_fn = sample["image_id"] + image_extension
        image_path = Path(image_root) / image_fn
        image_paths.append(image_path)

    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    return list(map(str, image_paths))


def _apply_prompt_template(
    prompt_template: str,
    context: str,
    add_answer_in_one_word_prompt: bool = False,
    dataspec_name: str | None = None,
) -> str:
    prompt = prompt_template.format(context)
    if add_answer_in_one_word_prompt and dataspec_name in ["m5b_vgr", "xvnli", "marvl"]:
        prompt += "\n Answer using only a single world."
    return prompt


def _store_sample_outputs(
    output_data: dict,
    output_dataspec_root: Path,
    dataspecs_name: str,
    verbose: bool = False,
) -> None:
    output_ds_p = output_dataspec_root / f"{dataspecs_name}.jsonl"
    df = pd.DataFrame(output_data)
    df.to_json(output_ds_p, orient="records", lines=True, force_ascii=False)
    if verbose:
        logger.info(f"Saved {len(df)} output data to: {output_ds_p}")


def get_all_sample_outputs_if_exists(
    output_dataspec_root: Path,
    dataspecs_name: str,
    ds_size: int,
) -> pd.DataFrame | None:
    output_ds_p = output_dataspec_root / f"{dataspecs_name}.jsonl"
    if output_ds_p.exists():
        df = pd.read_json(output_ds_p, orient="records", lines=True)
        if len(df) == ds_size:
            return df
    return None


def generate_sample_outputs(
    add_answer_in_one_word_prompt: bool,
    model: BaselineModel,
    dataspec_name: str,
    image_extension: str,
    image_root: Path,
    prompt_template: str,
    sample: dict[str, str],
) -> tuple[list[str], str, list[str]]:
    image_paths = _get_image_paths(
        sample=sample,
        image_root=image_root,
        image_extension=image_extension,
    )
    prompt = _apply_prompt_template(
        prompt_template=prompt_template,
        context=sample["context"],
        add_answer_in_one_word_prompt=add_answer_in_one_word_prompt,
        dataspec_name=dataspec_name,
    )

    output_text = model.generate_text(prompt=prompt, image_paths=image_paths)

    return image_paths, prompt, output_text


def main(
    hf_model_id: str,
    ex_allow: list[str] | None = None,
    ex_deny: list[str] | None = None,
    cfg_root_p: Path | str = "configs",
    data_root: Path | str = "/ltstorage/home/7schneid/gitrepos/mblipv2/data",
    images_root: Path | str = "/ltstorage/shares/datasets/mblipv2/images",
    output_root: Path
    | str = "/ltstorage/home/7schneid/gitrepos/mblipv2/results/mblipv2_test_suite/baseline_eval",
    yes: bool = False,
    only_eval: bool = False,
    add_answer_in_one_word_prompt: bool = False,
    max_failed_samples: int | float = 0.1,
) -> None:
    seed_everything(1337)

    cfg_root_p = Path(cfg_root_p)
    data_root = Path(data_root)
    images_root = Path(images_root)
    output_root = Path(output_root) / f"{hf_model_id.replace('/', '__')}"

    # stupid bug from Fire
    if isinstance(ex_allow, str):
        if ex_allow[0] == "[" and ex_allow[-1] == "]":
            ex_allow = ex_allow[1:-1].split(",")
    # ex_allow = ast.literal_eval(ex_allow)
    if isinstance(ex_deny, str):
        if ex_deny[0] == "[" and ex_deny[-1] == "]":
            ex_deny = ex_deny[1:-1].split(",")
        # ex_deny = ast.literal_eval(ex_deny)

    if not cfg_root_p.exists():
        raise FileNotFoundError(f"Config root not found: {cfg_root_p}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    logger.info("Running baseline evaluation...")
    logger.info(f"Model ID: {hf_model_id}")
    logger.info(f"Datasets allow: {ex_allow}")
    logger.info(f"Datasets ignore: {ex_deny}")
    logger.info(f"Config root: {cfg_root_p}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Images root: {images_root}")

    logger.info("Finding all test suite experiment configs...")
    test_suite_ex_cfgs = get_all_test_suite_ex_cfgs(
        cfg_root_p=cfg_root_p, ex_allow=ex_allow, ex_deny=ex_deny
    )
    logger.info(
        f"Found {len(test_suite_ex_cfgs)} test suite experiment configs:\n{pformat(test_suite_ex_cfgs)}"
    )

    if not yes:
        y_n = input("Continue? [y/n]: ")
        if y_n.lower() != "y":
            logger.info("Exiting...")
            return

    model = None
    if not only_eval:
        logger.info("Loading model...")
        model = load_model(hf_model_id=hf_model_id)

    logger.info("Loading datasets for each experiment...")

    # LOOP OVER TEST SUITE EXPERIMENT CONFIGS --> DATASPEC AKA DATASET
    for test_suite_ex_cfg_p in tqdm(test_suite_ex_cfgs, position=0, leave=True):
        experiment_dataspecs = load_test_suite_experiment_dataspecs(
            test_suite_ex_cfg_p=test_suite_ex_cfg_p,
            cfg_root_p=cfg_root_p,
            data_root=data_root,
            images_root=images_root,
        )
        logger.info(
            f"Datasets for experiment {test_suite_ex_cfg_p}: {len(experiment_dataspecs)}"
        )

        logger.info(pformat(experiment_dataspecs))  # for debugging

        # LOOP OVER DATASPECS OF THE DATASPEC AKA DATASET --> LANGUAGES SPLITS
        eval_results = []
        output_dataspec_root = None
        for dataspecs_config in experiment_dataspecs:
            ds = dataspecs_config["dataset"]
            dataspecs_name = dataspecs_config["dataspecs_name"]
            dataspec_name = dataspecs_config["dataspec_name"]
            image_extension = dataspecs_config["image_extension"]
            image_root = dataspecs_config["image_root"]
            prompt_template = dataspecs_config["prompt_template"]

            max_failed_samples_per_dataspec = 0
            if isinstance(max_failed_samples, float):
                max_failed_samples_per_dataspec = int(len(ds) * max_failed_samples)
            if isinstance(max_failed_samples, int):
                max_failed_samples_per_dataspec = max_failed_samples

            output_dataspec_root = output_root.with_name(
                output_root.name + "__" + dataspec_name
            )
            if not output_dataspec_root.exists():
                output_dataspec_root.mkdir(parents=True)

            # CHECK IF ALL SAMPLE OUTPUTS ALREADY EXIST (e.g., for re-evaluation or crash)
            sample_outputs_df = get_all_sample_outputs_if_exists(
                output_dataspec_root, dataspecs_name, len(ds)
            )
            if sample_outputs_df is not None:
                logger.info(
                    f"Found all sample outputs for {dataspecs_name} with {len(ds)} samples"
                )
                try:
                    eval_result = run_eval(
                        dataspec_name=dataspec_name,
                        dataspecs_name=dataspecs_name,
                        sample_outputs=sample_outputs_df,
                        output_dataspec_root=output_dataspec_root,
                        data_root=data_root,
                    )
                    eval_result["dataspec_name"] = dataspec_name
                    eval_result["dataspecs_name"] = dataspecs_name
                    eval_result["model_id"] = hf_model_id
                    eval_results.append(eval_result)
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                continue

            if model is None:
                raise ValueError("Model is None, but only_eval is False")

            # outputs of the dataspecs, i.e., individual samples
            sample_outputs = {
                "model_id": [],
                "dataspecs_name": [],
                "dataspec_name": [],
                "prompt": [],
                "output_text": [],
                "label": [],
                "text_label": [],
                "image_id": [],
                "pair_id": [],
                "context": [],
                "image_paths": [],
            }

            failed_samples = []

            # LOOP OVER THE DATASET OF THE DATASPECS TO GENERATE OUTPUTS --> SAMPLES
            for sample in tqdm(
                ds,
                position=1,
                leave=False,
                total=len(ds),
                desc=f"{hf_model_id} --> {dataspecs_name}",
            ):
                try:
                    image_paths, prompt, output_text = generate_sample_outputs(
                        model=model,
                        sample=sample,
                        prompt_template=prompt_template,
                        dataspec_name=dataspec_name,
                        image_root=image_root,
                        image_extension=image_extension,
                        add_answer_in_one_word_prompt=add_answer_in_one_word_prompt,
                    )
                except Exception as e:
                    logger.error(
                        f"Error during sample outputs generation for sample {sample.get('image_id', sample.get('pair_id', sample.get('context', 'n/a')))}: {e}"
                    )
                    failed_samples.append(sample)
                    if len(failed_samples) > max_failed_samples_per_dataspec:
                        logger.error(
                            f"Too many failed samples ({len(failed_samples)})! Exiting..."
                        )
                        logger.error(f"Failed samples: {failed_samples}")
                        raise SystemExit(
                            f"Too many failed samples ({len(failed_samples)})! Exiting..."
                        )
                else:
                    sample_outputs["model_id"].append(hf_model_id)
                    sample_outputs["dataspecs_name"].append(dataspecs_name)
                    sample_outputs["dataspec_name"].append(dataspec_name)
                    sample_outputs["prompt"].append(prompt)
                    sample_outputs["output_text"].append(output_text)
                    sample_outputs["label"].append(sample.get("label", "N/A - NOT SET"))
                    sample_outputs["text_label"].append(
                        sample.get("text_label", "N/A - NOT SET")
                    )
                    sample_outputs["image_id"].append(
                        sample.get("image_id", "N/A - NOT SET")
                    )
                    sample_outputs["pair_id"].append(
                        sample.get("pair_id", "N/A - NOT SET")
                    )
                    sample_outputs["context"].append(
                        sample.get("context", "N/A - NOT SET")
                    )
                    sample_outputs["image_paths"].append(image_paths)

                    if len(sample_outputs["model_id"]) % 100 == 0:
                        _store_sample_outputs(
                            output_data=sample_outputs,
                            output_dataspec_root=output_dataspec_root,
                            dataspecs_name=dataspecs_name,
                            verbose=False,
                        )

            _store_sample_outputs(
                output_data=sample_outputs,
                output_dataspec_root=output_dataspec_root,
                dataspecs_name=dataspecs_name,
                verbose=False,
            )

            sample_outputs_df = pd.DataFrame(sample_outputs)

            # RUN EVALUATION
            try:
                eval_result = run_eval(
                    dataspec_name=dataspec_name,
                    dataspecs_name=dataspecs_name,
                    sample_outputs=sample_outputs_df,
                    output_dataspec_root=output_dataspec_root,
                    data_root=data_root,
                )
                eval_result["dataspec_name"] = dataspec_name
                eval_result["dataspecs_name"] = dataspecs_name
                eval_result["model_id"] = hf_model_id
                eval_results.append(eval_result)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                continue

        if len(eval_results) == len(experiment_dataspecs):
            # only save combined eval results if all dataspecs were evaluated
            eval_results_df = pd.DataFrame(eval_results)
            eval_results_fn = (
                output_dataspec_root / f"{dataspec_name}_eval_results.json"
            )
            eval_results_df.to_json(
                eval_results_fn,
                orient="records",
            )
            logger.info(
                f"Saved all eval for {dataspec_name} results to: {eval_results_fn}"
            )


if __name__ == "__main__":
    Fire(main)
