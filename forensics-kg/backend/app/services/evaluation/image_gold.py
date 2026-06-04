"""Ground-truth adapters for image-analysis evaluation.

Each forensic image type has a different ground-truth source:
  - ballistics / tool_marks : synthetic gold JSON with known parameters
  - document_forensics      : CEDAR filename (original_* = genuine, forgeries_* = forged)
  - wound                   : AZH category dir (BG = no wound, else wound present)
  - bloodstain              : Attinger dataset — all impact-beating spatter (mechanism GT)
  - fingerprint             : SOCOFing filename encodes hand + finger position

Adapters expose, per type:
  * discover(limit)          -> [{image_path, gold:{attr:value}}]
  * specs                    -> [AttrSpec] describing how to score each attribute
  * predicted(result)        -> {attr:value} pulled from the pipeline's analyze() output
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional


@dataclass
class AttrSpec:
    name: str
    kind: str               # "categorical" | "binary" | "numeric"
    tolerance: float = 0.0   # numeric: |pred-gold| <= tolerance counts as correct
    note: str = ""


def _flatten_predicted(result: Dict[str, Any]) -> Dict[str, Any]:
    """Merge analyze() metadata + all entity properties into one flat dict."""
    flat: Dict[str, Any] = {}
    flat.update(result.get("metadata", {}) or {})
    for ent in result.get("entities", []) or []:
        for k, v in (ent.get("properties", {}) or {}).items():
            flat.setdefault(k, v)
    return flat


def cat_match(pred: Any, gold: Any) -> bool:
    """Fuzzy categorical match: normalized substring or high similarity."""
    if pred is None or gold is None:
        return False
    p = re.sub(r"[^a-z0-9 ]", " ", str(pred).lower()).strip()
    g = re.sub(r"[^a-z0-9 ]", " ", str(gold).lower()).strip()
    if not p or not g:
        return False
    if p in g or g in p:
        return True
    # token overlap (e.g. ".357 magnum" vs "357 mag")
    pt, gt = set(p.split()), set(g.split())
    if pt & gt:
        return True
    return SequenceMatcher(None, p, g).ratio() >= 0.7


class ImageGoldAdapter(ABC):
    image_type: str = ""
    specs: List[AttrSpec] = []

    def __init__(self, settings):
        self._settings = settings
        self._data_dir = Path(settings.data_dir)

    @abstractmethod
    def discover(self, limit: int) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def predicted(self, result: Dict[str, Any]) -> Dict[str, Any]:
        ...


# ── Ballistics ─────────────────────────────────────────────────────────────
class BallisticsGold(ImageGoldAdapter):
    image_type = "ballistics"
    specs = [
        AttrSpec("caliber", "categorical", note="firearm caliber vs synthetic gold"),
    ]

    def discover(self, limit: int) -> List[Dict[str, Any]]:
        img_dir = self._data_dir / "images" / "ballistics"
        gold_dir = self._data_dir / "gold_ballistics"
        out = []
        for img in sorted(img_dir.glob("*.png"))[:limit]:
            gp = gold_dir / f"{img.stem}.json"
            if gp.exists():
                g = json.loads(gp.read_text())
                out.append({"image_path": img, "gold": {
                    "caliber": g.get("caliber"),
                    "striation_count": g.get("striation_count"),
                }})
        return out

    def predicted(self, result):
        f = _flatten_predicted(result)
        return {"caliber": f.get("caliber") or f.get("caliber_estimate"),
                "striation_count": f.get("striation_count")}


# ── Tool marks ─────────────────────────────────────────────────────────────
class ToolMarksGold(ImageGoldAdapter):
    image_type = "tool_marks"
    specs = [
        AttrSpec("tool_type", "categorical", note="tool that made the mark vs synthetic gold"),
    ]

    def discover(self, limit):
        img_dir = self._data_dir / "images" / "tool_marks"
        gold_dir = self._data_dir / "gold_toolmarks"
        out = []
        for img in sorted(img_dir.glob("*.png"))[:limit]:
            gp = gold_dir / f"{img.stem}.json"
            if gp.exists():
                g = json.loads(gp.read_text())
                out.append({"image_path": img, "gold": {
                    "tool_type": g.get("tool_type"),
                    "striation_count": g.get("striation_count"),
                }})
        return out

    def predicted(self, result):
        f = _flatten_predicted(result)
        return {"tool_type": f.get("tool_class") or f.get("mark_type") or f.get("tool_type"),
                "striation_count": f.get("striation_count")}


# ── Document forensics (CEDAR) ─────────────────────────────────────────────
class DocumentGold(ImageGoldAdapter):
    image_type = "document_forensics"
    specs = [AttrSpec("is_forged", "binary", note="forgery detection (genuine vs forged)")]

    def discover(self, limit):
        base = self._data_dir / "images" / "document_forensics"
        files = sorted(base.rglob("original_*.png")) + sorted(base.rglob("forgeries_*.png"))
        # interleave genuine/forged so a small sample has both classes
        gen = [f for f in files if f.name.startswith("original_")]
        forg = [f for f in files if f.name.startswith("forgeries_")]
        out = []
        for i in range(max(len(gen), len(forg))):
            if i < len(gen):
                out.append({"image_path": gen[i], "gold": {"is_forged": False}})
            if i < len(forg):
                out.append({"image_path": forg[i], "gold": {"is_forged": True}})
            if len(out) >= limit:
                break
        return out[:limit]

    def predicted(self, result):
        f = _flatten_predicted(result)
        val = f.get("is_forged")
        if val is None:
            # infer from forgery indicators / document_type text
            txt = str(f.get("document_type", "")) + " " + str(f.get("authenticity", ""))
            has_forgery_entity = any(
                e.get("entity_type") == "ForgeryIndicator"
                for e in result.get("entities", [])
            )
            val = has_forgery_entity or "forg" in txt.lower()
        return {"is_forged": bool(val)}


# ── Wound (AZH) — detection only (clinical classes != forensic types) ──────
class WoundGold(ImageGoldAdapter):
    image_type = "wound"
    specs = [AttrSpec("wound_present", "binary", note="wound detected (AZH BG vs non-BG)")]

    def discover(self, limit):
        base = self._data_dir / "images" / "wounds" / "azh" / "Test"
        out = []
        # BG = background/no-wound; other category dirs = wound present
        cats = [d for d in base.iterdir() if d.is_dir()] if base.exists() else []
        bg = [d for d in cats if d.name.upper() == "BG"]
        nonbg = [d for d in cats if d.name.upper() not in ("BG", "TEST")]
        pool = []
        for d in bg:
            for img in sorted(d.glob("*"))[:max(1, limit // 2)]:
                if img.is_file():
                    pool.append((img, False))
        for d in nonbg:
            for img in sorted(d.glob("*"))[:2]:
                if img.is_file():
                    pool.append((img, True))
        for img, wound in pool[:limit]:
            out.append({"image_path": img, "gold": {"wound_present": wound}})
        return out

    def predicted(self, result):
        f = _flatten_predicted(result)
        present = bool(f.get("wound_detected", False)) or any(
            e.get("entity_type") == "WoundPattern" for e in result.get("entities", [])
        )
        return {"wound_present": present}


# ── Bloodstain (Attinger — all impact spatter) ─────────────────────────────
class BloodstainGold(ImageGoldAdapter):
    image_type = "bloodstain"
    specs = [AttrSpec("is_impact_spatter", "binary", note="Attinger impact-spatter recall")]

    def discover(self, limit):
        out = []
        dirs = getattr(self._settings, "resolved_bloodstain_dirs", [])
        for base in dirs:
            base = Path(base)
            if not base.exists():
                continue
            for sub in sorted(base.iterdir()):
                if sub.is_dir():
                    imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.JPG"))
                    if imgs:
                        out.append({"image_path": imgs[0], "gold": {"is_impact_spatter": True}})
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        return out[:limit]

    def predicted(self, result):
        f = _flatten_predicted(result)
        txt = " ".join(str(f.get(k, "")) for k in ("pattern_type", "mechanism", "mechanism_type"))
        return {"is_impact_spatter": ("impact" in txt.lower() or "spatter" in txt.lower())}


# ── Fingerprint (SOCOFing filename: <id>__<gender>_<hand>_<finger>_finger) ──
class FingerprintGold(ImageGoldAdapter):
    image_type = "fingerprint"
    specs = [
        AttrSpec("hand", "categorical", note="left/right (SOCOFing label)"),
        AttrSpec("finger", "categorical", note="thumb/index/.../little (SOCOFing label)"),
    ]
    _re = re.compile(r"__([MF])_(Left|Right)_(\w+?)_finger", re.IGNORECASE)

    def discover(self, limit):
        base = self._data_dir / "images" / "fingerprints"
        # Prefer the unaltered "Real" set; sample DIVERSE (hand, finger) combos so
        # the metric isn't dominated by repeated variants of one finger.
        real = next((p for p in base.rglob("Real") if p.is_dir()), None)
        search_root = real or base
        seen_combos: set = set()
        out: List[Dict[str, Any]] = []
        files = sorted(search_root.glob("*.BMP")) or sorted(search_root.rglob("*.BMP"))
        for img in files:
            m = self._re.search(img.name)
            if not m:
                continue
            hand, finger = m.group(2).lower(), m.group(3).lower()
            combo = (hand, finger)
            if combo in seen_combos:
                continue
            seen_combos.add(combo)
            out.append({"image_path": img, "gold": {"hand": hand, "finger": finger}})
            if len(out) >= limit:
                break
        return out

    def predicted(self, result):
        f = _flatten_predicted(result)
        return {"hand": f.get("hand"), "finger": f.get("finger_position") or f.get("finger")}


ADAPTERS: Dict[str, type] = {
    "ballistics": BallisticsGold,
    "tool_marks": ToolMarksGold,
    "document_forensics": DocumentGold,
    "wound": WoundGold,
    "bloodstain": BloodstainGold,
    "fingerprint": FingerprintGold,
}


def get_image_adapter(image_type: str, settings) -> Optional[ImageGoldAdapter]:
    cls = ADAPTERS.get(image_type)
    return cls(settings) if cls else None
