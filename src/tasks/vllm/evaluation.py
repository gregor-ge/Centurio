import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from tqdm import tqdm
import torch
import unicodedata
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
# from torchmetrics.functional.text import rouge_score
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from torchvision.ops import box_area
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

from src.tasks.vllm.data import get_tokenizer_llava


# Validation Split Train Loss
def output_loss(trident_module, outputs: dict, *args, **kwargs) -> dict:
    outputs["loss"] = outputs["loss"].detach().unsqueeze(0)
    return outputs

def validation_loss(loss):
    return torch.mean(loss)


def output_logits(trident_module, outputs: dict, *args, **kwargs) -> dict:
    image_token_index = trident_module.model.model.config.image_token_index
    input_ids = kwargs["batch"]["input_ids"]
    attention_mask = kwargs["batch"]["attention_mask"]
    logits = outputs["logits"].detach()

    # recreate llava insertion into input ids and attention mask:
    special_image_token_mask = input_ids == image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    # hardcoding 1 image for now
    num_image_patches = (logits.size(1) - kwargs["batch"]["attention_mask"].size(1) + num_special_image_tokens)[0] // 1 #num_special_image_tokens

    batch_indices, non_image_indices = torch.where(input_ids != image_token_index)
    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    final_attention_mask = torch.zeros(
        logits.size(0), logits.size(1), dtype=attention_mask.dtype, device=logits.device
    )
    final_input_ids = torch.zeros(
        logits.size(0), logits.size(1), dtype=attention_mask.dtype, device=logits.device
    )
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(logits.device),
        non_image_indices.to(logits.device),
        text_to_overwrite.to(logits.device),
    )
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]

    # compute length normalized log likelihood
    masked_log_probs = final_attention_mask.unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
    seq_token_log_probs = torch.gather(masked_log_probs[:, :-1, :], -1, final_input_ids[:, 1:].unsqueeze(-1)).squeeze()  # shift for next token
    mean_log_probs = seq_token_log_probs.sum(dim=-1) / final_attention_mask.sum(dim=-1)
    outputs["log_probs"] = mean_log_probs
    return outputs


# Caption Generation
def set_generation_mode(trident_module, batch, split=None, dataset_name=None, mode="generate", *args, **kwargs):
    batch["mode"] = mode
    batch["generate_kwargs"] = kwargs
    return batch

class OutputGenerate:
    def __init__(self, tokenizer):
        self.tokenizer = get_tokenizer_llava(tokenizer)
        self.split_generate = True
        self.model_name = tokenizer

    def __call__(self, trident_module, outputs=None, **kwargs) -> dict:

        # <|assistant|> is a special token for Phi-3 so we have no non-special token to split on.
        if "Phi-3" in self.model_name:
            captions_special = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            captions_special = [c.split("<|assistant|>")[1].strip() for c in captions_special]
            caption_ids = self.tokenizer(captions_special, add_special_tokens=False)["input_ids"]
            captions = self.tokenizer.batch_decode(caption_ids, skip_special_tokens=True)
        # Unlike normal Mistral, Nemo has [/INST] as a special token.
        elif "Mistral-Nemo" in self.model_name:
            captions_special = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            captions_special = [c.split("[/INST]")[1].strip() for c in captions_special]
            caption_ids = self.tokenizer(captions_special, add_special_tokens=False)["input_ids"]
            captions = self.tokenizer.batch_decode(caption_ids, skip_special_tokens=True)
        else:
            captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            if self.split_generate and "ASSISTANT" in captions[0]:
                captions = [c.split("ASSISTANT:")[1].strip() for c in captions]
            if self.split_generate and "[/INST]" in captions[0]:
                captions = [c.split("[/INST]")[1].strip() for c in captions]
            if self.split_generate and "<|assistant|>" in captions[0]:
                captions = [c.split("<|assistant|>")[1].strip() for c in captions]
            if self.split_generate and "assistant\n" in captions[0]:
                captions = [c.split("assistant\n")[1].strip() for c in captions]
            if self.split_generate and "model\n" in captions[0]:
                captions = [c.split("model\n")[1].strip() for c in captions]
            if self.split_generate and "<|CHATBOT_TOKEN|>" in captions[0]:
                captions = [c.split("<|CHATBOT_TOKEN|>")[1].strip() for c in captions]
        output = dict(caption=captions)
        return output


def noop_eval(image_ids, text_labels, captions, print_examples=10, dump_results=None):

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    if dump_results:
        json.dump([captions, image_ids, text_labels], open(dump_results, "w"))

    for i in range(min(print_examples, len(captions))):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")

    return dict(noop=0.0)
    # return acc



class MyCOCOEvalCap(COCOEvalCap):
    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

def caption_evaluation(annotation_file, image_ids, captions, text_labels, print_examples=10, dump_results=None,
                       filter_bbox=True):

    results = dict()

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        captions = [c for caps in captions for c in caps]
        text_labels = [l for labels in text_labels for l in labels]

    if dump_results:
        json.dump([captions, image_ids, text_labels], open(dump_results, "w"))

    if isinstance(text_labels[0], str) and text_labels[0] in {"ja", "th", "zh"}:
        if text_labels[0] == "zh":
            from spacy.lang.zh import Chinese
            chinese = Chinese() #.from_config({"nlp": {"tokenizer": {"segmenter": "jieba"}}})
            captions = [" ".join([word.text for word in chinese(caption)]) for caption in captions]
        if text_labels[0] == "ja":
            from spacy.lang.ja import Japanese
            japanese = Japanese()
            captions = [" ".join([word.text for word in japanese(caption)]) for caption in captions]
        if text_labels[0] == "th":
            from spacy.lang.th import Thai
            thai = Thai()
            captions = [" ".join([word.text for word in thai(caption)]) for caption in captions]
        # tokenizer = AutoTokenizer.from_pretrained("facebook/xlm-v-base")
        # captions = [" ".join(tokenizer.tokenize(c))[1:] for c in captions]
    captions = [unicodedata.normalize("NFC", c) for c in captions]

    for i in range(min(print_examples, len(captions))):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]}")

    if filter_bbox:
        bbox_pattern = r'\[.*?\]'
        captions = [re.sub(bbox_pattern, '', cap) for cap in captions]

    prediction = [dict(image_id=imgid, caption=caption) for imgid, caption in zip (image_ids, captions)]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as temp_file:
        json.dump(prediction, temp_file)

    # For template annotation files, we take encode in text_label which file to load (not clean but works)
    if "{" in annotation_file:
        annotation_file = annotation_file.format(text_labels[0])
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(temp_file.name)

    # create coco_eval object by taking coco and coco_result
    coco_eval = MyCOCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()
    os.unlink(temp_file.name)
    results = dict()
    for metric, score in coco_eval.eval.items():
        results[metric] = score
    return results

def classification_evaluation(image_ids, text_labels, captions, print_examples=10, vqa_process=True, log_prefix=False, dump_results=None):

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    if vqa_process:
        captions = vqa_clean(captions)


    if dump_results:
        json.dump([captions, image_ids, text_labels], open(dump_results, "w"))

    for i in range(min(print_examples, len(captions))):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")

    try:
        logging.info(f"Unique predictions: {len(set(captions))}. Unique prefix: {len(set([c.split()[0] for c in captions]))}")
        logging.info(f"Prefixes {set([c.split()[0] for c in captions])}")
        logging.info(f"Count: Total {len(captions)} | {Counter([c.split()[0] for c in captions])}")
        prefix_results = Counter([c.split()[0] for c in captions])
        prefix_results = {k: v/len(captions) for k,v in prefix_results.items()}
    except:
        prefix_results = dict()
        pass
    correct = 0
    total = 0
    relaxed_correct = 0
    for label, caption in zip(text_labels, captions):
        label = label.lower().strip()
        caption = caption.strip().lower()

        if label == caption:
            correct += 1
        if caption.startswith(label) or caption.endswith(label):
            relaxed_correct += 1
        total += 1
    acc = correct/total
    relaxed_acc = relaxed_correct/total
    results = dict(acc=acc, relaxed_acc=relaxed_acc)
    if log_prefix:
        results.update(prefix_results)
    return results
    # return acc


def vqa_classification_evaluation(image_ids, text_labels, captions, print_examples=10, vqa_process=True, dump_results=None):

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    if vqa_process:
        captions = vqa_clean(captions)

    for i in range(min(print_examples, len(captions))):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")


    if dump_results:
        json.dump([captions, image_ids, text_labels], open(dump_results, "w"))

    correct = 0
    total = 0
    relaxed_correct = 0
    for labels, caption in zip(text_labels, captions):
        labels = [l.lower().strip() for l in labels]
        caption = caption.strip().lower()
        for i in range(len(labels)):
            other_labels = [labels[j] for j in range(len(labels)) if j!=i]
            hits = len([1 for label in other_labels if label==caption])
            relaxed_hits = len([1 for label in other_labels if caption.startswith(label)])
            correct += min(1, float(hits/3.0))
            relaxed_correct += min(1, float(relaxed_hits/3.0))
            total += 1
    acc = correct/total
    relaxed_acc = relaxed_correct/total
    return dict(acc=acc, relaxed_acc=relaxed_acc)

def vqa_maxm_classification_evaluation(image_ids, text_labels, captions, print_examples=10, vqa_process=False, dump_results=None):

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    if vqa_process:
        captions = vqa_clean(captions)


    if dump_results:
        json.dump([captions, image_ids, text_labels], open(dump_results, "w"))

    for i in range(min(print_examples, len(captions))):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")

    try:
        logging.info(f"Unique predictions: {len(set(captions))}. Unique prefix: {len(set([c.split()[0] for c in captions]))}")
        logging.info(f"Prefixes {set([c.split()[0] for c in captions])}")
    except:
        pass
    correct = 0
    total = 0
    for labels, caption in zip(text_labels, captions):
        labels = [l.lower().strip() for l in labels]
        caption = caption.strip().lower()
        hits = len([1 for label in labels if label==caption])
        correct += min(1, hits)
        total += 1
    acc = correct/total

    # rouge = rouge_score(captions, text_labels, rouge_keys=("rougeL"))

    # return dict(acc=acc, rouge_l=rouge["rougeL_fmeasure"])

    return acc


def mme_eval(image_ids, text_labels, captions, print_examples=2):
    eval_type_dict = {
        "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark",
                       "artwork", "OCR"],
        "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
    }
    task = eval_type_dict["Perception"] + eval_type_dict["Cognition"]

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    # group by task
    text_labels = [l.split(" ### ") for l in text_labels]
    tasks, labels = [l[0] for l in text_labels], [l[1] for l in text_labels]
    task_results = {task: [[], [], []] for task in tasks}

    for task, label, caption, image_id in zip(tasks, labels, captions, image_ids):
        task_results[task][0].append(image_id)
        task_results[task][1].append(label)
        task_results[task][2].append(caption)

    task_metrics = defaultdict(lambda : defaultdict(lambda : 0))

    for task, (t_imgs, t_labels, t_caps) in task_results.items():
        for i in range(min(print_examples, len(captions))):
            logging.info(f"task: {task} -- img id: {t_imgs[i]} -- {t_caps[i]} --- Label: {t_labels[i]}")
        try:
            logging.info(
                f"Unique predictions: {len(set(t_caps))}. Unique prefix: {len(set([c.split()[0] for c in t_caps]))}")
            logging.info(f"Prefixes {set([c.split()[0] for c in t_caps])}")
        except:
            pass
        acc = 0
        acc_plus = 0
        for i in range(0, len(t_imgs), 2):
            img1, img2 = t_imgs[i:i+2]
            l1, l2 = t_labels[i:i+2]
            c1, c2 = t_caps[i:i+2]
            assert img1 == img2
            correct1 = int(c1.lower().startswith(l1.lower()))
            correct2 = int(c2.lower().startswith(l2.lower()))
            acc += correct1 + correct2
            acc_plus += (correct1+correct2) // 2
        acc = 100 * acc/len(t_imgs)
        acc_plus = 100 * acc_plus / (len(t_imgs)*0.5)
        total = acc + acc_plus
        task_metrics[task] = dict(acc=acc, acc_plus=acc_plus, score=total)

    sum_perception = sum([task_metrics[t]["acc"] for t in eval_type_dict["Perception"]]+[task_metrics[t]["acc_plus"] for t in eval_type_dict["Perception"]])
    sum_cognition = sum([task_metrics[t]["acc"] for t in eval_type_dict["Cognition"]]+[task_metrics[t]["acc_plus"] for t in eval_type_dict["Cognition"]])

    results = {f"{task}/{metric}": task_metrics[task][metric] for metric in task_metrics[task] for task in task_metrics}
    results["total_perception"] = sum_perception
    results["total_cognition"] = sum_cognition

    return results



def mmbench_eval(image_ids, text_labels, captions, print_examples=4):

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    # group by categories
    text_labels = [l.split(" ### ") for l in text_labels]
    l2category, category, labels = [l[0] for l in text_labels], [l[1] for l in text_labels], [l[2] for l in text_labels]
    cat_results = {cat: [[], [], []] for cat in l2category}

    for cat, label, caption, image_id in zip(l2category, labels, captions, image_ids):
        cat_results[cat][0].append(image_id)
        cat_results[cat][1].append(label)
        cat_results[cat][2].append(caption)

    cat_metrics = dict()

    total_acc = 0
    total_count = 0

    for cat, (t_imgs, t_labels, t_caps) in cat_results.items():
        for i in range(min(print_examples, len(captions))):
            logging.info(f"task: {cat} -- img id: {t_imgs[i]} -- {t_caps[i]} --- Label: {t_labels[i]}")
        try:
            logging.info(
                f"Unique predictions: {len(set(t_caps))}. Unique prefix: {len(set([c.split()[0] for c in t_caps]))}")
            logging.info(f"Prefixes {set([c.split()[0] for c in t_caps])}")
        except:
            pass
        acc = 0
        total = 0

        cur_image = t_imgs[0]
        correct = True
        for img, label, caption in zip (t_imgs, t_labels, t_caps):

            if img != cur_image:
                total += 1
                total_count += 1
                if correct:
                    acc += 1
                    total_acc += 1
                correct = True
                cur_image = img

            if not caption.lower().startswith(label.lower()):
                correct = False
        # once more at end for last round
        total += 1
        total_count += 1
        if correct:
            acc += 1
            total_acc += 1

        acc = 100 * acc/total
        cat_metrics[cat] = dict(acc=acc)

    results = {f"{cat}/acc": cat_metrics[cat]["acc"] for cat in cat_metrics}
    results["total/acc"] = 100 * total_acc/total_count

    return results



def mc_cycle_eval(image_ids, text_labels, captions, print_examples=4):

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    # group by categories
    text_labels = [l.split("###") for l in text_labels]
    labels, question_ids = [l[0] for l in text_labels], [l[1] for l in text_labels]

    acc = 0
    total = 0

    for i in range(min(print_examples, len(captions))):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {labels[i]}")
    try:
        logging.info(
            f"Unique predictions: {len(set(captions))}. Unique prefix: {len(set([c.split()[0] for c in captions]))}")
        logging.info(f"Prefixes {set([c.split()[0] for c in captions])}")
    except:
        pass

    cur_q = question_ids[0]
    correct = True
    for q_id, label, caption in zip (question_ids, labels, captions):

        if q_id != cur_q:
            total += 1
            if correct:
                acc += 1
            correct = True
            cur_q = q_id

        if not caption.lower().startswith(label.lower()):
            correct = False
    # once more at end for last round
    total += 1
    if correct:
        acc += 1

    results = dict()
    results["total/acc"] = 100 * acc/total

    return results


def sugarcrepe_eval(image_ids, text_labels, log_probs, print_examples=10):


    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]

    type_counter = defaultdict(lambda : 0)
    type_acc = defaultdict(lambda : 0)
    total_counter = 0
    total_acc = 0

    for i in range(0, len(image_ids)-1, 2):
        pos_iid = image_ids[i]
        neg_iid = image_ids[i+1]
        pos_label, pos_type = text_labels[i].split(" ### ")
        neg_label, neg_type = text_labels[i+1].split(" ### ")

        assert pos_label == "pos" and neg_label == "neg" and pos_iid == neg_iid

        pos_log = log_probs[i]
        neg_log = log_probs[i+1]

        if i < print_examples:
            print(f"{pos_iid}\t{pos_log}\t{neg_log}")

        correct = int(pos_log > neg_log)
        type_counter[pos_type] += 1
        type_acc[pos_type] += correct
        total_counter += 1
        total_acc += correct

    results = dict()
    for type in type_acc.keys():
        results[f"{type}"] = type_acc[type] / type_counter[type]
    results["total_acc"] = total_acc / total_counter
    return results


def log_probs_eval(image_ids, text_labels, log_probs, num_examples=4, print_examples=10):
    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]

    total_counter = 0
    total_acc = 0

    num_neg_examples = num_examples-1

    for i in range(0, len(image_ids)-num_neg_examples, num_examples):
        pos_iid = image_ids[i]
        neg_iids = image_ids[i+1:i+num_examples]
        pos_label = text_labels[i]
        neg_labels = text_labels[i+1:i+num_examples]

        assert (pos_label == "pos" and all(neg_label == "neg" for neg_label in neg_labels)
                and all(pos_iid == neg_iid for neg_iid in neg_iids))

        pos_log = log_probs[i]
        neg_logs = log_probs[i+1:i+num_examples]

        if i < print_examples:
            print(f"{pos_iid}\t{pos_log}\t{neg_logs}")

        correct = int(pos_log > max(neg_logs))
        total_counter += 1
        total_acc += correct

    results = dict()
    results["acc"] = total_acc / total_counter
    return results

# based on https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/evaluate_grounding.py
def grounding_eval(image_ids, text_labels, captions, print_examples=10):

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    for i in range(min(print_examples, len(captions))):
        logging.info(f"img id: {image_ids[i]} -- {captions[i]} --- Label: {text_labels[i]}")


    PATTERN = re.compile(r'(0\.\d*)')
    total_count = 0
    correct = 0
    for caption, label in zip(captions, text_labels):
        predict_bbox = re.findall(PATTERN, caption)
        try:
            if len(predict_bbox) != 4:
                predict_bbox = [0., 0., 0., 0.]
            else:
                predict_bbox = [float(bbox) for bbox in predict_bbox]
        except:
            predict_bbox = [0., 0., 0., 0.]

        target_bbox = torch.tensor(json.loads(label), dtype=torch.float32).view(-1, 4)
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
        iou, _ = box_iou(predict_bbox, target_bbox)
        iou = iou.item()
        total_count += 1
        if iou >= 0.5:
            correct += 1
    return correct/total_count

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union



#adapted from https://github.com/salesforce/LAVIS/blob/main/lavis/common/vqa_tools/vqa_eval.py
def vqa_clean(captions):
    manualMap = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    articles = ["a", "an", "the"]
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(,)(\d)")
    punct = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]
    contractions = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = " ".join(outText)
        return outText

    cleaned_captions = []
    for cap in captions:
        cap = cap.replace("\n", "").replace("\t", "").strip()
        cap = processPunctuation(cap)
        cap = processDigitArticle(cap)
        cleaned_captions.append(cap)
    return cleaned_captions


chair_coco_object_synonyms = [['person', 'girl', 'boy', 'man', 'woman', 'kid', 'child', 'chef', 'baker', 'people', 'adult', 'rider', 'children', 'baby', 'worker', 'passenger', 'sister', 'biker', 'policeman', 'cop', 'officer', 'lady', 'cowboy', 'bride', 'groom', 'male', 'female', 'guy', 'traveler', 'mother', 'father', 'gentleman', 'pitcher', 'player', 'skier', 'snowboarder', 'skater', 'skateboarder', 'person', 'woman', 'guy', 'foreigner', 'child', 'gentleman', 'caller', 'offender', 'coworker', 'trespasser', 'patient', 'politician', 'soldier', 'grandchild', 'serviceman', 'walker', 'drinker', 'doctor', 'bicyclist', 'thief', 'buyer', 'teenager', 'student', 'camper', 'driver', 'solider', 'hunter', 'shopper', 'villager'], ['bicycle', 'bike', 'bicycle', 'bike', 'unicycle', 'minibike', 'trike'], ['car', 'automobile', 'van', 'minivan', 'sedan', 'suv', 'hatchback', 'cab', 'jeep', 'coupe', 'taxicab', 'limo', 'taxi'], ['motorcycle', 'scooter', ' motor bike', 'motor cycle', 'motorbike', 'scooter', 'moped'], ['airplane', 'jetliner', 'plane', 'air plane', 'monoplane', 'aircraft', 'jet', 'jetliner', 'airbus', 'biplane', 'seaplane'], ['bus', 'minibus', 'trolley'], ['train', 'locomotive', 'tramway', 'caboose'], ['truck', 'pickup', 'lorry', 'hauler', 'firetruck'], ['boat', 'ship', 'liner', 'sailboat', 'motorboat', 'dinghy', 'powerboat', 'speedboat', 'canoe', 'skiff', 'yacht', 'kayak', 'catamaran', 'pontoon', 'houseboat', 'vessel', 'rowboat', 'trawler', 'ferryboat', 'watercraft', 'tugboat', 'schooner', 'barge', 'ferry', 'sailboard', 'paddleboat', 'lifeboat', 'freighter', 'steamboat', 'riverboat', 'battleship', 'steamship'], ['traffic light', 'street light', 'traffic signal', 'stop light', 'streetlight', 'stoplight'], ['fire hydrant', 'hydrant'], ['stop sign'], ['parking meter'], ['bench', 'pew'], ['bird', 'ostrich', 'owl', 'seagull', 'goose', 'duck', 'parakeet', 'falcon', 'robin', 'pelican', 'waterfowl', 'heron', 'hummingbird', 'mallard', 'finch', 'pigeon', 'sparrow', 'seabird', 'osprey', 'blackbird', 'fowl', 'shorebird', 'woodpecker', 'egret', 'chickadee', 'quail', 'bluebird', 'kingfisher', 'buzzard', 'willet', 'gull', 'swan', 'bluejay', 'flamingo', 'cormorant', 'parrot', 'loon', 'gosling', 'waterbird', 'pheasant', 'rooster', 'sandpiper', 'crow', 'raven', 'turkey', 'oriole', 'cowbird', 'warbler', 'magpie', 'peacock', 'cockatiel', 'lorikeet', 'puffin', 'vulture', 'condor', 'macaw', 'peafowl', 'cockatoo', 'songbird'], ['cat', 'kitten', 'feline', 'tabby'], ['dog', 'puppy', 'beagle', 'pup', 'chihuahua', 'schnauzer', 'dachshund', 'rottweiler', 'canine', 'pitbull', 'collie', 'pug', 'terrier', 'poodle', 'labrador', 'doggie', 'doberman', 'mutt', 'doggy', 'spaniel', 'bulldog', 'sheepdog', 'weimaraner', 'corgi', 'cocker', 'greyhound', 'retriever', 'brindle', 'hound', 'whippet', 'husky'], ['horse', 'colt', 'pony', 'racehorse', 'stallion', 'equine', 'mare', 'foal', 'palomino', 'mustang', 'clydesdale', 'bronc', 'bronco'], ['sheep', 'lamb', 'ram', 'lamb', 'goat', 'ewe'], ['cow', 'cattle', 'oxen', 'ox', 'calf', 'cattle', 'holstein', 'heifer', 'buffalo', 'bull', 'zebu', 'bison'], ['elephant'], ['bear', 'panda'], ['zebra'], ['giraffe'], ['backpack', 'knapsack'], ['umbrella'], ['handbag', 'wallet', 'purse', 'briefcase'], ['tie', 'bow', 'bow tie'], ['suitcase', 'suit case', 'luggage'], ['frisbee'], ['skis', 'ski'], ['snowboard'], ['sports ball', 'ball'], ['kite'], ['baseball bat'], ['baseball glove'], ['skateboard'], ['surfboard', 'longboard', 'skimboard', 'shortboard', 'wakeboard'], ['tennis racket', 'racket'], ['bottle'], ['wine glass'], ['cup'], ['fork'], ['knife', 'pocketknife', 'knive'], ['spoon'], ['bowl', 'container'], ['banana'], ['apple'], ['sandwich', 'burger', 'sub', 'cheeseburger', 'hamburger'], ['orange'], ['broccoli'], ['carrot'], ['hot dog'], ['pizza'], ['donut', 'doughnut', 'bagel'], ['cake', ' cheesecake', 'cupcake', 'shortcake', 'coffeecake', 'pancake'], ['chair', 'seat', 'stool'], ['couch', 'sofa', 'recliner', 'futon', 'loveseat', 'settee', 'chesterfield'], ['potted plant', 'houseplant'], ['bed'], ['dining table', 'table', 'desk'], ['toilet', 'urinal', 'commode', 'toilet', 'lavatory', 'potty'], ['tv', 'monitor', 'televison', 'television'], ['laptop', 'computer', 'notebook', 'netbook', 'lenovo', 'macbook', 'laptop computer'], ['mouse'], ['remote'], ['keyboard'], ['cell phone', 'mobile phone', 'phone', 'cellphone', 'telephone', 'phon', 'smartphone', 'iPhone'], ['microwave'], ['oven', 'stovetop', 'stove', 'stove top oven'], ['toaster'], ['sink'], ['refrigerator', 'fridge', 'fridge', 'freezer'], ['book'], ['clock'], ['vase'], ['scissors'], ['teddy bear', 'teddybear'], ['hair drier', 'hairdryer'], ['toothbrush']]
def chair(image_ids, text_labels, captions, print_examples=10, filter_bbox=True, dump_results=None):
    import nltk
    nltk.download('punkt')
    # nltk.download('omw-1.4')


    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    if filter_bbox:
        bbox_pattern = r'\[.*?\]'
        captions = [re.sub(bbox_pattern, '', cap) for cap in captions]

    synonyms = chair_coco_object_synonyms
    mscoco_objects = []  # mscoco objects and *all* synonyms
    inverse_synonym_dict = {}
    for synonym in synonyms:
        mscoco_objects.extend(synonym)
        for s in synonym:
            inverse_synonym_dict[s] = synonym[0]

    # Some hard coded rules for implementing CHAIR metrics on MSCOCO

    # common 'double words' in MSCOCO that should be treated as a single word
    coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
                         'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter',
                         'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket',
                         'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
                         'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog',
                         'teddy bear', 'home plate', 'train track']

    # Hard code some rules for special cases in MSCOCO
    # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
    animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'animal', 'cub']
    # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
    vehicle_words = ['jet', 'train']

    # double_word_dict will map double words to the word they should be treated as in our analysis

    double_word_dict = {}
    for double_word in coco_double_words:
        double_word_dict[double_word] = double_word
    for animal_word in animal_words:
        double_word_dict['baby %s' % animal_word] = animal_word
        double_word_dict['adult %s' % animal_word] = animal_word
    for vehicle_word in vehicle_words:
        double_word_dict['passenger %s' % vehicle_word] = vehicle_word
    double_word_dict['bow tie'] = 'tie'
    double_word_dict['toilet seat'] = 'toilet'
    double_word_dict['wine glas'] = 'wine glass'
    def caption_to_words(caption):
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''

        # standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]

        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = ' '.join(words[i:i + 2])
            if double_word in double_word_dict:
                double_words.append(double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

        # get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(mscoco_objects)]
        words = [word for word in words if word in set(mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(inverse_synonym_dict[word])
        # return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    num_caps = 0.
    num_hallucinated_caps = 0.
    hallucinated_word_count = 0.
    coco_word_count = 0.
    coverages = []
    predictions = []
    objects = 0

    for i, (iid, cap, gt_objects) in enumerate(zip(image_ids, captions, text_labels)):
        gt_objects = set(gt_objects)

        # get all words in the caption, as well as corresponding node word
        words, node_words, idxs, raw_words = caption_to_words(cap)

        # count hallucinated words
        coco_word_count += len(node_words)
        hallucinated = False
        mscoco_hallucinated_words = []
        objects += len(node_words)
        coverage = set()
        for word, node_word, idx in zip(words, node_words, idxs):
            if node_word not in gt_objects:
                hallucinated_word_count += 1
                mscoco_hallucinated_words.append((word, node_word))
                hallucinated = True
            else:
                coverage.add(node_word)
        if len(gt_objects)>0:
            coverages.append(len(coverage)/len(gt_objects))
        if print_examples > 0 and i < print_examples:
            print(f"{iid} -- {cap} -- {words} {node_words} -- {mscoco_hallucinated_words}")
        predictions.append((iid, cap, words, node_words, mscoco_hallucinated_words))
        # count hallucinated caps
        num_caps += 1
        if hallucinated:
            num_hallucinated_caps += 1

        # cap_dict['metrics']['CHAIRs'] = int(hallucinated)
        # cap_dict['metrics']['CHAIRi'] = 0.
        # if len(words) > 0:
        #     cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words']) / float(len(words))

    average_objects = (objects/num_caps)
    average_coverage = sum(coverages)/len(coverages)
    average_words = sum([len(c.split()) for c in captions])/len(captions)
    chair_s = (num_hallucinated_caps / num_caps)
    chair_i = (hallucinated_word_count / coco_word_count) if coco_word_count > 0 else 0

    if dump_results:
        json.dump(predictions, open(dump_results, "w"))

    return dict(chair_s=chair_s, chair_i=chair_i, chair_avrg_words=average_words, chair_coverage=average_coverage, chair_objects=average_objects)



#### SINGULARIZE ###################################################################################
# Adapted from Bermi Ferrer's Inflector for Python:
# http://www.bermi.org/inflector/

# Copyright (c) 2006 Bermi Ferrer Martinez
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software to deal in this software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of this software, and to permit
# persons to whom this software is furnished to do so, subject to the following
# condition:
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THIS SOFTWARE.

singular_rules = [
    (r'(?i)(.)ae$'            , '\\1a'    ),
    (r'(?i)(.)itis$'          , '\\1itis' ),
    (r'(?i)(.)eaux$'          , '\\1eau'  ),
    (r'(?i)(quiz)zes$'        , '\\1'     ),
    (r'(?i)(matr)ices$'       , '\\1ix'   ),
    (r'(?i)(ap|vert|ind)ices$', '\\1ex'   ),
    (r'(?i)^(ox)en'           , '\\1'     ),
    (r'(?i)(alias|status)es$' , '\\1'     ),
    (r'(?i)([octop|vir])i$'   , '\\1us'  ),
    (r'(?i)(cris|ax|test)es$' , '\\1is'   ),
    (r'(?i)(shoe)s$'          , '\\1'     ),
    (r'(?i)(o)es$'            , '\\1'     ),
    (r'(?i)(bus)es$'          , '\\1'     ),
    (r'(?i)([m|l])ice$'       , '\\1ouse' ),
    (r'(?i)(x|ch|ss|sh)es$'   , '\\1'     ),
    (r'(?i)(m)ovies$'         , '\\1ovie' ),
    (r'(?i)(.)ombies$'        , '\\1ombie'),
    (r'(?i)(s)eries$'         , '\\1eries'),
    (r'(?i)([^aeiouy]|qu)ies$', '\\1y'    ),
        # -f, -fe sometimes take -ves in the plural
        # (e.g., lives, wolves).
    (r"([aeo]l)ves$"          , "\\1f"    ),
    (r"([^d]ea)ves$"          , "\\1f"    ),
    (r"arves$"                , "arf"     ),
    (r"erves$"                , "erve"    ),
    (r"([nlw]i)ves$"          , "\\1fe"   ),
    (r'(?i)([lr])ves$'        , '\\1f'    ),
    (r"([aeo])ves$"           , "\\1ve"   ),
    (r'(?i)(sive)s$'          , '\\1'     ),
    (r'(?i)(tive)s$'          , '\\1'     ),
    (r'(?i)(hive)s$'          , '\\1'     ),
    (r'(?i)([^f])ves$'        , '\\1fe'   ),
    # -ses suffixes.
    (r'(?i)(^analy)ses$'      , '\\1sis'  ),
    (r'(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$', '\\1\\2sis'),
    (r'(?i)(.)opses$'         , '\\1opsis'),
    (r'(?i)(.)yses$'          , '\\1ysis' ),
    (r'(?i)(h|d|r|o|n|b|cl|p)oses$', '\\1ose'),
    (r'(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$', '\\1ose'),
    (r'(?i)(.)oses$'          , '\\1osis' ),
    # -a
    (r'(?i)([ti])a$'          , '\\1um'   ),
    (r'(?i)(n)ews$'           , '\\1ews'  ),
    (r'(?i)s$'                , ''        ),
]

# For performance, compile the regular expressions only once:
singular_rules = [(re.compile(r[0]), r[1]) for r in singular_rules]

singular_uninflected = set((
    "bison"      , "debris"   , "headquarters", "pincers"    , "trout"     ,
    "bream"      , "diabetes" , "herpes"      , "pliers"     , "tuna"      ,
    "breeches"   , "djinn"    , "high-jinks"  , "proceedings", "whiting"   ,
    "britches"   , "eland"    , "homework"    , "rabies"     , "wildebeest",
    "carp"       , "elk"      , "innings"     , "salmon"     ,
    "chassis"    , "flounder" , "jackanapes"  , "scissors"   ,
    "christmas"  , "gallows"  , "mackerel"    , "series"     ,
    "clippers"   , "georgia"  , "measles"     , "shears"     ,
    "cod"        , "graffiti" , "mews"        , "species"    ,
    "contretemps",              "mumps"       , "swine"      ,
    "corps"      ,              "news"        , "swiss"      ,
))
singular_uncountable = set((
    "advice"     , "equipment", "happiness"   , "luggage"    , "news"      , "software"     ,
    "bread"      , "fruit"    , "information" , "mathematics", "progress"  , "understanding",
    "butter"     , "furniture", "ketchup"     , "mayonnaise" , "research"  , "water"        ,
    "cheese"     , "garbage"  , "knowledge"   , "meat"       , "rice"      ,
    "electricity", "gravel"   , "love"        , "mustard"    , "sand"      ,
))
singular_ie = set((
    "alergie"    , "cutie"    , "hoagie"      , "newbie"     , "softie"    , "veggie"       ,
    "auntie"     , "doggie"   , "hottie"      , "nightie"    , "sortie"    , "weenie"       ,
    "beanie"     , "eyrie"    , "indie"       , "oldie"      , "stoolie"   , "yuppie"       ,
    "birdie"     , "freebie"  , "junkie"      , "^pie"       , "sweetie"   , "zombie"       ,
    "bogie"      , "goonie"   , "laddie"      , "pixie"      , "techie"    ,
    "bombie"     , "groupie"  , "laramie"     , "quickie"    , "^tie"      ,
    "collie"     , "hankie"   , "lingerie"    , "reverie"    , "toughie"   ,
    "cookie"     , "hippie"   , "meanie"      , "rookie"     , "valkyrie"  ,
))
singular_irregular = {
       "atlantes": "atlas",
        "atlases": "atlas",
           "axes": "axe",
         "beeves": "beef",
       "brethren": "brother",
       "children": "child",
        "corpora": "corpus",
       "corpuses": "corpus",
    "ephemerides": "ephemeris",
           "feet": "foot",
        "ganglia": "ganglion",
          "geese": "goose",
         "genera": "genus",
          "genii": "genie",
       "graffiti": "graffito",
         "helves": "helve",
           "kine": "cow",
         "leaves": "leaf",
         "loaves": "loaf",
            "men": "man",
      "mongooses": "mongoose",
         "monies": "money",
          "moves": "move",
         "mythoi": "mythos",
         "numena": "numen",
       "occipita": "occiput",
      "octopodes": "octopus",
          "opera": "opus",
         "opuses": "opus",
            "our": "my",
           "oxen": "ox",
          "penes": "penis",
        "penises": "penis",
         "people": "person",
          "sexes": "sex",
    "soliloquies": "soliloquy",
          "teeth": "tooth",
         "testes": "testis",
        "trilbys": "trilby",
         "turves": "turf",
            "zoa": "zoon",
}

plural_prepositions = set((
    "about"  , "before" , "during", "of"   , "till" ,
    "above"  , "behind" , "except", "off"  , "to"   ,
    "across" , "below"  , "for"   , "on"   , "under",
    "after"  , "beneath", "from"  , "onto" , "until",
    "among"  , "beside" , "in"    , "out"  , "unto" ,
    "around" , "besides", "into"  , "over" , "upon" ,
    "at"     , "between", "near"  , "since", "with" ,
    "athwart", "betwixt",
               "beyond",
               "but",
               "by"))

VERB, NOUN, ADJECTIVE, ADVERB = "VB", "NN", "JJ", "RB"

def singularize(word, pos=NOUN, custom={}):
    """ Returns the singular of a given word.
    """
    if word in custom:
        return custom[word]
    # Recurse compound words (e.g. mothers-in-law).
    if "-" in word:
        w = word.split("-")
        if len(w) > 1 and w[1] in plural_prepositions:
            return singularize(w[0], pos, custom) + "-" + "-".join(w[1:])
    # dogs' => dog's
    if word.endswith("'"):
        return singularize(word[:-1]) + "'s"
    w = word.lower()
    for x in singular_uninflected:
        if x.endswith(w):
            return word
    for x in singular_uncountable:
        if x.endswith(w):
            return word
    for x in singular_ie:
        if w.endswith(x + "s"):
            return w
    for x in singular_irregular:
        if w.endswith(x):
            return re.sub('(?i)' + x + '$', singular_irregular[x], word)
    for suffix, inflection in singular_rules:
        m = suffix.search(word)
        g = m and m.groups() or []
        if m:
            for k in range(len(g)):
                if g[k] is None:
                    inflection = inflection.replace('\\' + str(k + 1), '')
            return suffix.sub(inflection, word)
    return word


def chair_embed(image_ids, text_labels, captions, print_examples=10, dump_results=None, filter_bbox=True,
                threshold_neg=0.78, threshold_pos=0.73, embedding_model="BAAI/bge-base-en-v1.5"):
    import spacy
    from sentence_transformers import SentenceTransformer

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    if filter_bbox:
        original_captions = captions
        bbox_pattern = r'\[.*?\]'
        captions = [re.sub(bbox_pattern, '', cap) for cap in captions]

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    noun_phrases = []
    for i, doc in tqdm(enumerate(nlp.pipe(captions)),
                       total=len(captions)):
        noun_phrases.append([chunk for chunk in doc.noun_chunks])


    embedding_model = SentenceTransformer(embedding_model)
    all_gt_objects = list(set(o for gt_os in text_labels for o in gt_os))
    object_embeddings = embedding_model.encode(all_gt_objects, normalize_embeddings=True, show_progress_bar=False)

    num_caps = 0.
    num_hallucinated_caps = 0.
    hallucinated_word_count = 0.
    coverages = []
    objects = 0


    predictions = []

    for i, (iid, cap, gt_objects, nps) in enumerate(zip(image_ids, captions, text_labels, noun_phrases)):
        if len(nps) > 0:
            np_embed = embedding_model.encode([np.text for np in nps], normalize_embeddings=True, show_progress_bar=False)

            if len(gt_objects) > 0:
                gt_embed = embedding_model.encode(gt_objects, normalize_embeddings=True, show_progress_bar=False)
                sims_pos = np_embed @ gt_embed.T
                gt_scores = sims_pos.max(axis=1)
                gt_match_idx = [(i, idx) for idx, (i, s) in enumerate(zip(sims_pos.argmax(axis=1), gt_scores)) if s >= threshold_pos]
                gt_matches = [gt_objects[i] for i, _ in gt_match_idx]
                gt_match_idx = [idx for _, idx in gt_match_idx]
            else:
                gt_match_idx = []
                gt_matches = []

            sims = np_embed @ object_embeddings.T
            scores = sims.max(axis=1)
            global_matches = [all_gt_objects[i] for idx, (i, s) in enumerate(zip(sims.argmax(axis=1), scores)) if
                              idx not in gt_match_idx and s >= threshold_neg and all_gt_objects[i] not in gt_objects]

            ignored_hallucinations = ["camera", "photo", "picture", "sign", "table"]
            if "grape" in global_matches and "wine" in cap: # matched wrongly
                ignored_hallucinations.append("grape")
            if "coffee machine" in global_matches and "cup" in gt_objects: # cup of coffee
                ignored_hallucinations.append("coffee")
            if "steak" in global_matches and "meat" in cap:  # meat matched to steak
                ignored_hallucinations.append("steak")
            if "balls" in global_matches and "billards" in gt_objects:  #sic
                ignored_hallucinations.append("balls")
            if "lettuce" in global_matches and "green vegetables" in gt_objects:
                ignored_hallucinations.append("lettuce")
            global_matches = [o for o in global_matches if all(ignored not in o for ignored in ignored_hallucinations)]
        else:
            gt_matches = []
            global_matches = []
        # count hallucinated words
        objects += len(gt_matches) + len(global_matches)
        hallucinated_word_count += len(global_matches)
        hallucinated = len(global_matches) > 0

        if len(gt_objects)>0:
            coverages.append(len(set(gt_matches))/len(gt_objects))
        if print_examples > 0 and i < print_examples:
            print(f"{iid} -- {cap} -- {gt_matches} -- {global_matches}")
        predictions.append([iid, cap, gt_matches, global_matches])
        if filter_bbox:
            predictions[-1].append(original_captions[i])
        # count hallucinated caps
        num_caps += 1
        if hallucinated:
            num_hallucinated_caps += 1

    average_objects = (objects/num_caps)
    average_coverage = sum(coverages)/len(coverages)
    average_words = sum([len(c.split()) for c in captions])/len(captions)
    chair_s = (num_hallucinated_caps / num_caps)
    chair_i = (hallucinated_word_count / objects) if objects > 0 else 0

    if dump_results:
        json.dump(predictions, open(dump_results, "w"))

    return dict(embed_chair_s=chair_s, embed_chair_i=chair_i, embed_chair_avrg_words=average_words,
                embed_chair_coverage=average_coverage, embed_chair_objects=average_objects)


def clip_score(image_ids, text_labels, captions, embedding_folder, print_examples=10, dump_results=None,
                embedding_model=('ViT-B-16-SigLIP-256', 'webli'), batchsize=512, filter_bbox=True):
    import open_clip
    import numpy as np
    from numpy.linalg import norm

    if isinstance(captions[0], list):
        image_ids = [id for ids in image_ids for id in ids]
        text_labels = [label for labels in text_labels for label in labels]
        captions = [c for caps in captions for c in caps]

    if filter_bbox:
        bbox_pattern = r'\[.*?\]'
        captions = [re.sub(bbox_pattern, '', cap) for cap in captions]

    image2idx = json.load(open(f"{embedding_folder}/image2idx.json"))
    image_embeddings = np.load(f"{embedding_folder}/image_embeddings.npy")

    model_name, pretrained = embedding_model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.cuda()

    scores = []
    text_embeddings = []
    for i in range(0, len(captions), batchsize):
        batch = tokenizer(captions[i:i+batchsize]).cuda()
        embs = model.encode_text(batch)
        text_embeddings.append(embs.cpu().numpy())
    text_embeddings = np.concatenate(text_embeddings, axis=0)

    for i, (iid, cap, text_embed) in enumerate(zip(image_ids, captions, text_embeddings)):
        img_embed = image_embeddings[image2idx[iid]]

        cos = max(0, np.dot(text_embed, img_embed)/(norm(text_embed)*norm(img_embed)))
        score = 100*cos
        scores.append(score)
        if i < print_examples:
            print(f"{iid} - {cap} - {score:.0f}")

    return np.mean(scores)
