from src.baseline_eval.models.baseline_model import BaselineModel


def get_all_supported_baseline_models() -> list[str]:
    return [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "microsoft/Phi-3.5-vision-instruct",
        "neulab/Pangea-7B-hf",
        "openbmb/MiniCPM-V-2_6",
        "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "mistralai/Pixtral-12B-2409",
        "AIDC-AI/Parrot-7B",
        "MBZUAI/PALO-7B",
        "MBZUAI/PALO-13B",
        "OpenGVLab/InternVL2_5-4B",
        "OpenGVLab/InternVL2_5-8B",
        "OpenGVLab/InternVL2_5-26B",
        "maya-multimodal/maya",
        "WueNLP/centurio_aya",
        "WueNLP/centurio_qwen",
    ]


def load_model(hf_model_id: str, device: str = "cuda") -> "BaselineModel":
    if hf_model_id in [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
    ]:
        from src.baseline_eval.models.qwen2_vl import Qwen2VL

        return Qwen2VL(hf_model_id=hf_model_id, device=device)
    elif hf_model_id == "microsoft/Phi-3.5-vision-instruct":
        from src.baseline_eval.models.phi3_5_vision import Phi3_5_Vision

        return Phi3_5_Vision(hf_model_id=hf_model_id, device=device)
    elif hf_model_id == "neulab/Pangea-7B-hf":
        from src.baseline_eval.models.pangea import Pangea

        return Pangea(hf_model_id=hf_model_id, device=device)
    elif hf_model_id == "openbmb/MiniCPM-V-2_6":
        from src.baseline_eval.models.minicpm_v_2_6 import MiniCPM_V_2_6

        return MiniCPM_V_2_6(hf_model_id=hf_model_id, device=device)
    elif hf_model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        from src.baseline_eval.models.llama3_2_vision import Llama3_2_Vision

        return Llama3_2_Vision(hf_model_id=hf_model_id, device=device)
    elif hf_model_id == "mistralai/Pixtral-12B-2409":
        from src.baseline_eval.models.pixtral import Pixtral

        return Pixtral(hf_model_id=hf_model_id)
    elif hf_model_id == "AIDC-AI/Parrot-7B":
        from src.parrot.inference.parrot import Parrot

        return Parrot(device=device)  # type: ignore

    elif hf_model_id in ["MBZUAI/PALO-7B", "MBZUAI/PALO-13B"]:
        from src.palo.inference.palo import PALO

        return PALO(hf_model_id=hf_model_id, device=device)  # type: ignore

    elif hf_model_id in [
        "OpenGVLab/InternVL2_5-4B",
        "OpenGVLab/InternVL2_5-8B",
        "OpenGVLab/InternVL2_5-26B",
    ]:
        from src.baseline_eval.models.internvl_2_5 import Intern_V_2_5

        return Intern_V_2_5(hf_model_id=hf_model_id, device=device)
    elif hf_model_id == "maya-multimodal/maya":
        from src.maya.llava.inference.maya import Maya

        return Maya(hf_model_id=hf_model_id, device=device)  # type: ignore
    elif hf_model_id in ["WueNLP/centurio_aya", "WueNLP/centurio_qwen"]:
        from src.baseline_eval.models.centurio import Centurio

        return Centurio(hf_model_id=hf_model_id, device=device)

    elif hf_model_id == "lmms-lab/llava-onevision-qwen2-7b-ov-chat":
        raise NotImplementedError(f"Model ID {hf_model_id} not yet implemented")
    else:
        raise KeyError(f"Model ID {hf_model_id} supported")


__all__ = ["load_model", "get_all_supported_baseline_models"]
