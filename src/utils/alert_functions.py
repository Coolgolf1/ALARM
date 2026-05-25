import json
import math
import pandas as pd
from time import perf_counter
from sklearn.metrics import accuracy_score, classification_report
from typing import Callable


def _to_records(data):
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    if isinstance(data, list):
        return data
    try:
        return list(data)
    except TypeError as exc:
        raise TypeError(
            "dataset must be a pandas DataFrame, list of dicts, or iterable of entries"
        ) from exc


def _to_binary_class(value) -> int:
    """Convert any Alert column value to 1 (alert) or 0 (no alert)."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0

    if isinstance(value, dict):
        if "is_alert" in value:
            return _parse_bool(value["is_alert"])
        return 1 if any(
            value.get(k) not in (None, "", "null")
            for k in ("alert_level", "category", "reason")
        ) else 0

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "null", "none", "nan"}:
            return 0
        try:
            return _to_binary_class(json.loads(stripped))
        except Exception:
            return 1

    return 1 if bool(value) else 0


def _parse_bool(value) -> int:
    """
    Safely convert an is_alert value to 0 or 1.

    Models sometimes output is_alert as the string "false" / "true" instead
    of a JSON boolean. bool("false") == True in Python, which would silently
    convert every no-alert prediction into an alert.  This function handles
    all realistic output formats explicitly.
    """
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return 1 if value else 0
    if isinstance(value, str):
        return 0 if value.strip().lower() in {"false", "0", "no", "null", "none", ""} else 1
    return 1 if value else 0


def _normalize_prediction(pred) -> dict:
    """Ensure prediction is a normalised dict with the four expected keys."""
    if hasattr(pred, "model_dump"):
        pred = pred.model_dump()
    elif isinstance(pred, str):
        pred = json.loads(pred)

    if not isinstance(pred, dict):
        raise TypeError(
            f"predict_fn must return dict, JSON string, or pydantic model — got {type(pred)}"
        )

    return {
        "is_alert":    bool(_parse_bool(pred.get("is_alert", False))),
        "alert_level": pred.get("alert_level"),
        "category":    pred.get("category"),
        "reason":      pred.get("reason"),
    }


def _compute_field_metrics(predictions: list, records: list, label_column: str) -> dict:
    """
    Compute accuracy for alert_level and category fields,
    evaluated only on entries where ground truth is an alert.
    """
    import math

    level_true, level_pred = [], []
    cat_true,   cat_pred = [], []

    for pred, entry in zip(predictions, records):
        raw = entry.get(label_column)

        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        if isinstance(raw, str) and raw.strip().lower() in {"null", "none", "nan", ""}:
            continue

        try:
            gt = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue

        if not isinstance(gt, dict):
            continue

        gt_level = gt.get("alert_level")
        gt_cat = gt.get("category")

        if gt_level is not None:
            level_true.append(str(gt_level))
            level_pred.append(str(pred.get("alert_level"))
                              if pred.get("alert_level") else "null")

        if gt_cat is not None:
            cat_true.append(str(gt_cat))
            cat_pred.append(str(pred.get("category"))
                            if pred.get("category") else "null")

    result = {}

    if level_true:
        correct = sum(t == p for t, p in zip(level_true, level_pred))
        result["alert_level"] = {
            "accuracy": correct / len(level_true),
            "n":        len(level_true),
            "report":   classification_report(
                level_true, level_pred, output_dict=True, zero_division=0
            ),
        }

    if cat_true:
        correct = sum(t == p for t, p in zip(cat_true, cat_pred))
        result["category"] = {
            "accuracy": correct / len(cat_true),
            "n":        len(cat_true),
            "report":   classification_report(
                cat_true, cat_pred, output_dict=True, zero_division=0
            ),
        }

    return result


def benchmark_alert_model(
    predict_fn: Callable[[dict], dict],
    dataset,
    label_column: str = "Alert",
) -> dict:
    """
    Run predict_fn over every entry in dataset and compute binary metrics.

    Parameters
    ----------
    predict_fn   : callable(entry: dict) -> dict
                   Must return a dict with at least the key ``is_alert``.
    dataset      : pd.DataFrame, list of dicts, or any iterable of dicts
    label_column : column / key that holds the ground-truth Alert value

    Returns
    -------
    dict with keys:
        predictions            – list of normalised prediction dicts
        metrics                – mean latency, mean output tokens, n_entries
        classification_metrics – accuracy + sklearn report (None if no labels)
    """
    records = _to_records(dataset)
    predictions = []
    y_true, y_pred, times, token_counts = [], [], [], []

    for entry in records:
        t0 = perf_counter()
        raw = predict_fn(entry)
        elapsed = perf_counter() - t0

        pred = _normalize_prediction(raw)
        predictions.append(pred)
        times.append(elapsed)
        token_counts.append(len(json.dumps(pred, ensure_ascii=True).split()))

        y_pred.append(1 if pred["is_alert"] else 0)
        if isinstance(entry, dict) and label_column in entry:
            y_true.append(_to_binary_class(entry[label_column]))

    n_pred_alert = sum(y_pred)
    n_pred_no_alert = len(y_pred) - n_pred_alert
    print(f"Predicted  : {n_pred_alert} alert / {n_pred_no_alert} no_alert")
    if y_true:
        n_true_alert = sum(y_true)
        n_true_no_alert = len(y_true) - n_true_alert
        print(
            f"Ground truth: {n_true_alert} alert / {n_true_no_alert} no_alert")

    result = {
        "predictions": predictions,
        "metrics": {
            "n_entries":          len(records),
            "mean_time_seconds":  sum(times) / len(times) if times else 0.0,
            "mean_output_tokens": sum(token_counts) / len(token_counts) if token_counts else 0.0,
        },
        "classification_metrics": None,
    }

    if len(y_true) == len(y_pred) and y_true:
        result["classification_metrics"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "report": classification_report(
                y_true,
                y_pred,
                labels=[1, 0],
                target_names=["alert", "no_alert"],
                output_dict=True,
                zero_division=0,
            ),
        }

    result["field_metrics"] = _compute_field_metrics(
        predictions, records, label_column)

    return result
