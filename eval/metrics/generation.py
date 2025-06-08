import evaluate
import numpy as np
from typing import List, Dict, Union, Optional

try:
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
except Exception as e:
    print(f"Warning: Evaluation metric could not be loaded: {e}")
    bleu = rouge = meteor = bertscore = None


def compute_text_generation_metrics(
        predictions: List[str],
        references: List[Union[str, List[str]]],
        device: Optional[str] = None,
        include_bertscore: bool = False
) -> Dict[str, float]:
    str_references = []
    list_references = []

    for ref in references:
        if isinstance(ref, list):
            str_references.append(ref[0])
            list_references.append(ref)
        else:
            str_references.append(ref)
            list_references.append([ref])

    metrics = {}
    bleu_metrics = compute_bleu_score(predictions, list_references)
    metrics.update(bleu_metrics)

    rouge_metrics = compute_rouge_scores(predictions, str_references)
    metrics.update(rouge_metrics)

    meteor_metrics = compute_meteor_score(predictions, str_references)
    metrics.update(meteor_metrics)

    if include_bertscore:
        bertscore_metrics = compute_bertscore(predictions, str_references, device=device)
        metrics.update(bertscore_metrics)

    return metrics


def compute_bleu_score(
        predictions: List[str],
        references: List[Union[str, List[str]]],
        max_order: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score for text generation.
    """
    if bleu is None:
        raise RuntimeError("BLEU metric not available")

    # ensure references is in the correct format (list of lists)
    if references and isinstance(references[0], str):
        references = [[ref] for ref in references]

    try:
        bleu_result = bleu.compute(
            predictions=predictions,
            references=references,
            max_order=max_order
        )

        return {
            "bleu": bleu_result["bleu"],
            "bleu_1": bleu_result.get("precisions", [0, 0, 0, 0])[0] if "precisions" in bleu_result else 0.0,
            "bleu_2": bleu_result.get("precisions", [0, 0, 0, 0])[1] if "precisions" in bleu_result else 0.0,
            "bleu_3": bleu_result.get("precisions", [0, 0, 0, 0])[2] if "precisions" in bleu_result else 0.0,
            "bleu_4": bleu_result.get("precisions", [0, 0, 0, 0])[3] if "precisions" in bleu_result else 0.0,
        }
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}


def compute_rouge_scores(
        predictions: List[str],
        references: List[str],
        rouge_types: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute ROUGE scores for text generation.
    """
    if rouge is None:
        raise RuntimeError("ROUGE metric not available")

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    try:
        rouge_result = rouge.compute(
            predictions=predictions,
            references=references,
            rouge_types=rouge_types
        )

        return {
            "rouge1": rouge_result.get("rouge1", 0.0),
            "rouge2": rouge_result.get("rouge2", 0.0),
            "rougeL": rouge_result.get("rougeL", 0.0),
            "rougeLsum": rouge_result.get("rougeLsum", 0.0),
        }
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}


def compute_meteor_score(
        predictions: List[str],
        references: List[str]
) -> Dict[str, float]:
    """
    Compute METEOR score for text generation.
    """
    if meteor is None:
        raise RuntimeError("METEOR metric not available")

    try:
        meteor_result = meteor.compute(predictions=predictions, references=references)
        return {"meteor": meteor_result["meteor"]}
    except Exception as e:
        print(f"Error computing METEOR: {e}")
        return {"meteor": 0.0}


def compute_bertscore(
        predictions: List[str],
        references: List[str],
        model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute BERTScore for text generation.
    """
    if bertscore is None:
        print("Warning: BERTScore not available, skipping")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}

    try:
        bertscore_result = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type=model_type,
            device=device
        )

        return {
            "bertscore_precision": np.mean(bertscore_result["precision"]),
            "bertscore_recall": np.mean(bertscore_result["recall"]),
            "bertscore_f1": np.mean(bertscore_result["f1"]),
        }
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
