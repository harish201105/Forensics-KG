"""Quantitative evaluation of the forensic image-analysis pipelines.

Runs the real analysis pipeline on labelled images and scores predicted
attributes against ground truth:
  * categorical / binary -> accuracy (+ precision/recall/F1 for binary)
  * numeric              -> mean absolute error + within-tolerance accuracy
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from app.services.extraction.image_extractor import ImageExtractor
from app.services.evaluation.image_gold import get_image_adapter, cat_match, ADAPTERS


class ImageEvaluator:
    def __init__(self, image_extractor: ImageExtractor, settings):
        self._images = image_extractor
        self._settings = settings

    async def evaluate(
        self, image_type: str, limit: int = 5, model: Optional[str] = None
    ) -> Dict[str, Any]:
        adapter = get_image_adapter(image_type, self._settings)
        if adapter is None:
            raise ValueError(f"No evaluation adapter for image type '{image_type}'")

        items = adapter.discover(limit)
        if not items:
            return {"image_type": image_type, "samples": 0,
                    "error": "No labelled images found for this type."}

        # accumulators per attribute
        acc: Dict[str, Dict[str, Any]] = {
            s.name: {"spec": s, "n": 0, "correct": 0, "abs_err": [],
                     "tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for s in adapter.specs
        }
        per_image: List[Dict[str, Any]] = []

        for idx, item in enumerate(items):
            img_path = item["image_path"]
            try:
                image_bytes = img_path.read_bytes()
                # IMPORTANT: use a LABEL-FREE id. The filename stem encodes ground
                # truth (e.g. "100__M_Left_index_finger", "forgeries_10_1"), so
                # passing it as metadata would leak the answer into the prompt.
                result = await self._images.analyze(
                    image_bytes, {"case_id": f"eval_{image_type}_{idx}"}, image_type=image_type
                )
            except Exception as e:
                logger.warning(f"Image eval failed for {img_path.name}: {e}")
                continue

            pred = adapter.predicted(result)
            gold = item["gold"]
            row: Dict[str, Any] = {"image": img_path.name, "attrs": {}}

            for s in adapter.specs:
                p, g = pred.get(s.name), gold.get(s.name)
                if g is None:
                    continue
                a = acc[s.name]
                a["n"] += 1
                if s.kind == "numeric":
                    try:
                        err = abs(float(p) - float(g))
                        a["abs_err"].append(err)
                        correct = err <= s.tolerance
                    except (TypeError, ValueError):
                        correct = False
                    if correct:
                        a["correct"] += 1
                    row["attrs"][s.name] = {"pred": p, "gold": g, "correct": correct}
                elif s.kind == "binary":
                    pb, gb = bool(p), bool(g)
                    correct = pb == gb
                    if correct:
                        a["correct"] += 1
                    if gb and pb:
                        a["tp"] += 1
                    elif gb and not pb:
                        a["fn"] += 1
                    elif (not gb) and pb:
                        a["fp"] += 1
                    else:
                        a["tn"] += 1
                    row["attrs"][s.name] = {"pred": pb, "gold": gb, "correct": correct}
                else:  # categorical
                    correct = cat_match(p, g)
                    if correct:
                        a["correct"] += 1
                    row["attrs"][s.name] = {"pred": p, "gold": g, "correct": correct}
            per_image.append(row)

        # finalize metrics
        attributes: Dict[str, Any] = {}
        for name, a in acc.items():
            s = a["spec"]
            n = a["n"]
            entry: Dict[str, Any] = {"kind": s.kind, "samples": n, "note": s.note}
            if n == 0:
                attributes[name] = {**entry, "accuracy": None}
                continue
            entry["accuracy"] = round(a["correct"] / n, 3)
            if s.kind == "numeric":
                errs = a["abs_err"]
                entry["mae"] = round(sum(errs) / len(errs), 2) if errs else None
                entry["within_tolerance"] = entry.pop("accuracy")
                entry["tolerance"] = s.tolerance
            elif s.kind == "binary":
                tp, fp, fn = a["tp"], a["fp"], a["fn"]
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                entry.update({"precision": round(prec, 3), "recall": round(rec, 3),
                              "f1": round(f1, 3),
                              "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": a["tn"]}})
            attributes[name] = entry

        return {
            "image_type": image_type,
            "samples": len(per_image),
            "model": model or getattr(self._settings, "openai_model", "default"),
            "attributes": attributes,
            "per_image": per_image,
        }

    @staticmethod
    def supported_types() -> List[str]:
        return list(ADAPTERS.keys())
