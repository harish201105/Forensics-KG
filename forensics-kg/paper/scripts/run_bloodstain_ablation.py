#!/usr/bin/env python3
"""Bloodstain image ablation: CV-only vs LLM-only vs CV+LLM (the hybrid pipeline).

All Attinger samples are impact-spatter, so the metric is detection of the
impact-spatter pattern (gold all-positive): recall = detection rate, precision
= 1.0 if any detection, F1 = 2R/(1+R). Writes bloodstain_ablation.json.
  CV-only  : classical CV features only (no LLM) -> no pattern label.
  LLM-only : vision model on the raw image, no CV preprocessing/conditioning.
  CV+LLM   : the full hybrid pipeline (CV features condition the LLM synthesis).
"""
import asyncio, json, sys, base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.image_extractor import ImageExtractor
from app.services.extraction.image_strategies.bloodstain import BloodstainStrategy
from app.services.extraction.schemas import BLOODSTAIN_IMAGE_ANALYSIS_SCHEMA
from app.services.evaluation.image_gold import BloodstainGold

settings = get_settings()


def score(preds):
    tp = sum(1 for p in preds if p)
    fn = sum(1 for p in preds if not p)
    prec = 1.0 if tp else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": len(preds), "detected": tp, "precision": round(prec, 2),
            "recall": round(rec, 2), "f1": round(f1, 2)}


async def main(limit=8):
    oc = OpenAIClient(settings)
    ie = ImageExtractor(oc, settings)
    strat = BloodstainStrategy(settings)
    adapter = BloodstainGold(settings)
    items = adapter.discover(limit)
    print(f"bloodstain samples discovered: {len(items)}", flush=True)
    if not items:
        print("NO bloodstain data found; cannot run ablation."); return

    cv_preds, llm_preds, full_preds = [], [], []
    for i, it in enumerate(items):
        b = it["image_path"].read_bytes()
        try:
            # CV-only: classical features, no LLM -> no pattern label
            cv = await strat.preprocess(b)
            cv_preds.append(adapter.predicted({"metadata": cv, "entities": []})["is_impact_spatter"])
            # LLM-only: vision on the raw image, no CV conditioning
            resized = ie._resize_for_api(b)
            img_b64 = base64.b64encode(resized).decode("utf-8")
            vp = await strat.vision_analysis(img_b64, {})
            vision = await oc.analyze_image(
                system_prompt=vp["system_prompt"], user_prompt=vp["user_prompt"],
                image_base64=img_b64, response_schema=BLOODSTAIN_IMAGE_ANALYSIS_SCHEMA)
            llm_preds.append(adapter.predicted({"metadata": vision, "entities": []})["is_impact_spatter"])
            # CV+LLM: full hybrid pipeline
            full = await ie.analyze(b, {"case_id": f"abl_{i}"}, image_type="bloodstain")
            full_preds.append(adapter.predicted(full)["is_impact_spatter"])
            print(f"  {it['image_path'].name}: cv={cv_preds[-1]} llm={llm_preds[-1]} full={full_preds[-1]}", flush=True)
        except Exception as e:
            print(f"  skip {it['image_path'].name}: {e}", flush=True)

    out = {"samples": len(full_preds),
           "CV_only": score(cv_preds),
           "LLM_only": score(llm_preds),
           "CV_LLM_hybrid": score(full_preds)}
    (Path(__file__).parent / "bloodstain_ablation.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    asyncio.run(main(lim))
