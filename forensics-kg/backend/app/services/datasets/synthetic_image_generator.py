"""Synthetic forensic image generator using OpenCV.

Generates procedural forensic images for training/testing when real datasets
are unavailable:
- Ballistics: bullet striation patterns, cartridge case impressions, rifling marks
- Tool marks: impression patterns, striation marks on surfaces
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class BallisticsImageGenerator:
    """Generate synthetic ballistics evidence images.

    Creates realistic-looking images of:
    - Bullet striation patterns (land-and-groove impressions)
    - Cartridge case head stamps and firing pin impressions
    - Rifling mark comparisons
    """

    def __init__(self, output_dir: Path, data_dir: Optional[Path] = None):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        root = data_dir or output_dir.parent.parent  # data/images/ballistics -> data/
        self._gold_dir = root / "gold_ballistics"
        self._gold_dir.mkdir(parents=True, exist_ok=True)

    def generate_batch(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic ballistics images with metadata."""
        results = []
        generators = [
            self._generate_bullet_striation,
            self._generate_cartridge_headstamp,
            self._generate_rifling_comparison,
        ]

        for i in range(count):
            gen_func = generators[i % len(generators)]
            image_type = ["bullet_striation", "cartridge_headstamp", "rifling_comparison"][i % 3]
            img_id = f"BAL-{i+1:04d}"

            img, metadata = gen_func(img_id, i)
            metadata["image_id"] = img_id
            metadata["image_type"] = image_type

            # Save image
            img_path = self._output_dir / f"{img_id}.png"
            cv2.imwrite(str(img_path), img)

            # Save gold metadata
            gold_path = self._gold_dir / f"{img_id}.json"
            gold_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            results.append(metadata)
            logger.debug(f"Generated ballistics image {img_id} ({image_type})")

        logger.info(f"Generated {len(results)} synthetic ballistics images")
        return results

    def _generate_bullet_striation(
        self, img_id: str, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a synthetic bullet surface with striation marks."""
        rng = np.random.RandomState(seed + 42)
        h, w = 512, 512

        # Base metallic surface
        img = np.full((h, w, 3), [160, 165, 170], dtype=np.uint8)

        # Add metallic noise texture
        noise = rng.normal(0, 8, (h, w, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Generate striation lines (parallel grooves)
        num_striations = rng.randint(15, 40)
        striation_data = []
        angle = rng.uniform(-5, 5)  # slight angle variation

        for j in range(num_striations):
            x_pos = int(w * (j + 0.5) / num_striations + rng.normal(0, 3))
            width = rng.randint(1, 4)
            depth = rng.randint(30, 90)  # darkness = depth indicator

            # Draw striation line
            x1 = int(x_pos + h * math.tan(math.radians(angle)) / 2)
            x2 = int(x_pos - h * math.tan(math.radians(angle)) / 2)
            color = int(max(60, 170 - depth))
            cv2.line(img, (x1, 0), (x2, h), (color, color + 3, color + 5), width)

            striation_data.append({
                "position_x": x_pos,
                "width_px": width,
                "relative_depth": round(depth / 90, 2),
            })

        # Add circular curvature effect (bullet is cylindrical)
        for y in range(h):
            for x in range(w):
                dist = abs(x - w // 2) / (w // 2)
                darken = int(40 * dist * dist)
                img[y, x] = np.clip(img[y, x].astype(np.int16) - darken, 0, 255)

        # Add subtle gaussian blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        # Add measurement scale bar
        cv2.rectangle(img, (20, h - 40), (120, h - 35), (255, 255, 255), -1)
        cv2.putText(img, "1mm", (45, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        caliber = random.choice(["9mm", ".45 ACP", ".38 Special", "7.62mm", ".22 LR", ".357 Magnum"])
        twist_dir = random.choice(["left", "right"])
        num_lands = random.choice([4, 5, 6, 8])

        metadata = {
            "caliber": caliber,
            "twist_direction": twist_dir,
            "num_lands_and_grooves": num_lands,
            "striation_count": num_striations,
            "striations": striation_data[:5],  # first 5 for gold
            "surface_condition": random.choice(["good", "moderate", "corroded"]),
            "magnification": random.choice(["10x", "20x", "40x"]),
        }
        return img, metadata

    def _generate_cartridge_headstamp(
        self, img_id: str, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a synthetic cartridge case headstamp image."""
        rng = np.random.RandomState(seed + 100)
        h, w = 512, 512

        # Dark background
        img = np.full((h, w, 3), [40, 40, 45], dtype=np.uint8)

        center = (w // 2, h // 2)
        radius = 180

        # Draw cartridge base (brass-colored circle)
        brass_color = (60, 170, 200)  # BGR for brass
        cv2.circle(img, center, radius, brass_color, -1)

        # Add concentric rings (tooling marks)
        for r in range(radius - 10, 30, -15):
            ring_color = tuple(int(c + rng.randint(-15, 15)) for c in brass_color)
            cv2.circle(img, center, r, ring_color, 1)

        # Firing pin impression (center)
        fp_radius = rng.randint(12, 25)
        fp_depth = rng.randint(3, 8)
        fp_shape = random.choice(["circular", "rectangular", "elliptical"])
        fp_center = (center[0] + rng.randint(-5, 5), center[1] + rng.randint(-5, 5))

        if fp_shape == "circular":
            cv2.circle(img, fp_center, fp_radius, (30, 100, 130), -1)
            cv2.circle(img, fp_center, fp_radius - 2, (40, 120, 150), -1)
        elif fp_shape == "rectangular":
            half = fp_radius
            cv2.rectangle(
                img,
                (fp_center[0] - half, fp_center[1] - half // 2),
                (fp_center[0] + half, fp_center[1] + half // 2),
                (30, 100, 130), -1,
            )
        else:
            cv2.ellipse(img, fp_center, (fp_radius, fp_radius // 2), 0, 0, 360, (30, 100, 130), -1)

        # Ejector mark
        ej_angle = rng.uniform(0, 2 * math.pi)
        ej_dist = radius - 40
        ej_pos = (
            int(center[0] + ej_dist * math.cos(ej_angle)),
            int(center[1] + ej_dist * math.sin(ej_angle)),
        )
        ej_size = rng.randint(8, 18)
        cv2.circle(img, ej_pos, ej_size, (40, 130, 160), -1)

        # Extractor mark
        ex_angle = ej_angle + math.pi + rng.uniform(-0.3, 0.3)
        ex_dist = radius - 30
        ex_pos = (
            int(center[0] + ex_dist * math.cos(ex_angle)),
            int(center[1] + ex_dist * math.sin(ex_angle)),
        )
        cv2.line(
            img,
            (ex_pos[0] - 10, ex_pos[1]),
            (ex_pos[0] + 10, ex_pos[1]),
            (50, 140, 170), 3,
        )

        # Headstamp text
        manufacturer = random.choice(["WIN", "REM", "FED", "PMC", "CCI", "HORT", "S&B"])
        cal_text = random.choice(["9MM", "45ACP", "38SPL", "762", "223"])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, manufacturer, (center[0] - 30, center[1] - 80), font, 0.5, (80, 190, 220), 1)
        cv2.putText(img, cal_text, (center[0] - 25, center[1] + 100), font, 0.5, (80, 190, 220), 1)

        # Add metallic noise
        noise = rng.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        metadata = {
            "manufacturer": manufacturer,
            "caliber_marking": cal_text,
            "firing_pin_shape": fp_shape,
            "firing_pin_diameter_px": fp_radius * 2,
            "ejector_mark_position_deg": round(math.degrees(ej_angle), 1),
            "ejector_mark_size_px": ej_size,
            "extractor_mark_position_deg": round(math.degrees(ex_angle), 1),
            "cartridge_diameter_px": radius * 2,
        }
        return img, metadata

    def _generate_rifling_comparison(
        self, img_id: str, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a side-by-side rifling comparison image."""
        rng = np.random.RandomState(seed + 200)
        h, w = 512, 1024

        # Two halves for comparison
        img = np.full((h, w, 3), [50, 50, 55], dtype=np.uint8)

        is_match = rng.random() > 0.4  # 60% chance of match
        num_striations = 25  # default, overwritten in loop

        for side in range(2):
            x_off = side * 512
            base_color = [155, 160, 165]
            half = np.full((h, 512, 3), base_color, dtype=np.uint8)

            # Base noise
            noise = rng.normal(0, 6, half.shape).astype(np.int16)
            half = np.clip(half.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Generate consistent striation pattern (or different if no match)
            base_seed = seed if (is_match or side == 0) else seed + 999
            pattern_rng = np.random.RandomState(base_seed + 300)
            num_striations = pattern_rng.randint(20, 35)

            for j in range(num_striations):
                x_pos = int(512 * (j + 0.5) / num_striations + pattern_rng.normal(0, 2))
                w_line = pattern_rng.randint(1, 3)
                depth = pattern_rng.randint(30, 80)

                # Add variation for non-matching pair
                if not is_match and side == 1:
                    x_pos += rng.randint(-8, 8)
                    depth = rng.randint(30, 80)

                color_val = int(max(70, 165 - depth))
                cv2.line(half, (x_pos, 0), (x_pos, h), (color_val, color_val + 2, color_val + 4), w_line)

            # Curvature
            for y in range(h):
                for x in range(512):
                    dist = abs(x - 256) / 256
                    darken = int(30 * dist * dist)
                    half[y, x] = np.clip(half[y, x].astype(np.int16) - darken, 0, 255)

            img[:, x_off:x_off + 512] = half

        # Divider line
        cv2.line(img, (512, 0), (512, h), (0, 200, 255), 2)

        # Labels
        cv2.putText(img, "Evidence", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
        cv2.putText(img, "Reference", (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)

        match_text = "MATCH" if is_match else "NO MATCH"
        match_color = (0, 255, 0) if is_match else (0, 0, 255)
        cv2.putText(img, match_text, (430, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, match_color, 2)

        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        metadata = {
            "comparison_result": "match" if is_match else "no_match",
            "evidence_caliber": random.choice(["9mm", ".45 ACP", ".38 Special", "7.62mm"]),
            "reference_caliber": random.choice(["9mm", ".45 ACP", ".38 Special", "7.62mm"]),
            "num_matching_striations": num_striations if is_match else rng.randint(2, 8),
            "confidence": round(rng.uniform(0.85, 0.99) if is_match else rng.uniform(0.1, 0.4), 2),
            "magnification": random.choice(["20x", "40x", "60x"]),
            "comparison_type": "land_impression",
        }
        return img, metadata


class ToolMarkImageGenerator:
    """Generate synthetic tool mark evidence images.

    Creates realistic-looking images of:
    - Striation marks from screwdrivers, pry bars, bolt cutters
    - Impression marks from hammers, pliers, wire cutters
    - Cut marks on various surfaces
    """

    def __init__(self, output_dir: Path, data_dir: Optional[Path] = None):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        root = data_dir or output_dir.parent.parent  # data/images/tool_marks -> data/
        self._gold_dir = root / "gold_toolmarks"
        self._gold_dir.mkdir(parents=True, exist_ok=True)

    def generate_batch(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic tool mark images with metadata."""
        results = []
        generators = [
            self._generate_striation_mark,
            self._generate_impression_mark,
            self._generate_cut_mark,
            self._generate_pry_mark,
        ]

        for i in range(count):
            gen_func = generators[i % len(generators)]
            mark_type = ["striation", "impression", "cut", "pry"][i % 4]
            img_id = f"TM-{i+1:04d}"

            img, metadata = gen_func(img_id, i)
            metadata["image_id"] = img_id
            metadata["mark_type"] = mark_type

            # Save image
            img_path = self._output_dir / f"{img_id}.png"
            cv2.imwrite(str(img_path), img)

            # Save gold metadata
            gold_path = self._gold_dir / f"{img_id}.json"
            gold_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            results.append(metadata)
            logger.debug(f"Generated tool mark image {img_id} ({mark_type})")

        logger.info(f"Generated {len(results)} synthetic tool mark images")
        return results

    def _make_surface(
        self, h: int, w: int, rng: np.random.RandomState, material: str
    ) -> np.ndarray:
        """Create a base surface texture for different materials."""
        if material == "metal":
            base = [170, 175, 180]
            noise_std = 6
        elif material == "wood":
            base = [90, 140, 180]
            noise_std = 12
        elif material == "plastic":
            base = [140, 140, 145]
            noise_std = 4
        else:  # painted
            color = random.choice([(180, 180, 200), (160, 200, 160), (200, 180, 160)])
            base = list(color)
            noise_std = 5

        img = np.full((h, w, 3), base, dtype=np.uint8)
        noise = rng.normal(0, noise_std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add grain for wood
        if material == "wood":
            for y in range(0, h, rng.randint(3, 8)):
                intensity = rng.randint(-20, 5)
                cv2.line(img, (0, y), (w, y + rng.randint(-2, 2)),
                         tuple(int(max(0, min(255, base[c] + intensity))) for c in range(3)), 1)

        return img

    def _generate_striation_mark(
        self, img_id: str, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate striation marks (e.g., from screwdriver sliding on metal)."""
        rng = np.random.RandomState(seed + 500)
        h, w = 512, 512
        material = random.choice(["metal", "painted"])

        img = self._make_surface(h, w, rng, material)

        # Create striation zone
        zone_y = h // 2 + rng.randint(-50, 50)
        zone_height = rng.randint(60, 150)
        zone_angle = rng.uniform(-10, 10)

        tool_type = random.choice(["screwdriver", "chisel", "pry_bar", "knife"])
        num_striations = rng.randint(20, 50)

        striation_details = []
        for j in range(num_striations):
            y_pos = zone_y - zone_height // 2 + int(zone_height * j / num_striations)
            y_pos += rng.randint(-2, 2)
            width = rng.randint(1, 3)
            depth = rng.randint(20, 70)

            x_start = rng.randint(50, 150)
            x_end = w - rng.randint(50, 150)

            y_offset = int((x_end - x_start) * math.tan(math.radians(zone_angle)))
            color_val = int(max(80, 180 - depth))
            cv2.line(img, (x_start, y_pos), (x_end, y_pos + y_offset),
                     (color_val, color_val + 2, color_val), width)

            if j < 5:
                striation_details.append({
                    "y_position": y_pos,
                    "width_px": width,
                    "relative_depth": round(depth / 70, 2),
                })

        # Add some edge damage around the striation zone
        for _ in range(rng.randint(3, 8)):
            cx = rng.randint(100, w - 100)
            cy = zone_y + rng.randint(-zone_height // 2 - 20, zone_height // 2 + 20)
            cv2.circle(img, (cx, cy), rng.randint(2, 6), (100, 105, 110), -1)

        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        # Scale bar
        cv2.rectangle(img, (20, h - 40), (120, h - 35), (255, 255, 255), -1)
        cv2.putText(img, "2mm", (45, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        metadata = {
            "tool_type": tool_type,
            "material": material,
            "striation_count": num_striations,
            "striation_zone_width_px": zone_height,
            "striation_angle_deg": round(zone_angle, 1),
            "striations": striation_details,
            "direction": random.choice(["left_to_right", "right_to_left"]),
            "force_estimate": random.choice(["light", "moderate", "heavy"]),
            "magnification": random.choice(["5x", "10x", "20x"]),
        }
        return img, metadata

    def _generate_impression_mark(
        self, img_id: str, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate impression marks (e.g., from hammer or pliers)."""
        rng = np.random.RandomState(seed + 600)
        h, w = 512, 512
        material = random.choice(["metal", "wood", "plastic"])

        img = self._make_surface(h, w, rng, material)

        tool = random.choice(["hammer", "pliers", "wrench", "vice_grips"])
        cx, cy = w // 2 + rng.randint(-30, 30), h // 2 + rng.randint(-30, 30)
        shape_desc = "unknown"

        # Create impression based on tool type
        if tool == "hammer":
            # Circular/oval depression
            rx = rng.randint(40, 80)
            ry = rng.randint(35, 75)
            angle = rng.uniform(0, 180)
            # Dark center (deeper)
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, (90, 95, 100), -1)
            # Lighter rim (raised edge)
            cv2.ellipse(img, (cx, cy), (rx + 5, ry + 5), angle, 0, 360, (200, 205, 210), 2)
            # Inner texture
            for _ in range(20):
                px = cx + rng.randint(-rx + 5, rx - 5)
                py = cy + rng.randint(-ry + 5, ry - 5)
                cv2.circle(img, (px, py), rng.randint(1, 3), (80 + rng.randint(0, 20),) * 3, -1)
            shape_desc = f"elliptical ({rx*2}x{ry*2} px)"

        elif tool == "pliers":
            # Two parallel jaw impressions
            gap = rng.randint(30, 60)
            jaw_w = rng.randint(15, 30)
            jaw_h = rng.randint(60, 120)
            angle = rng.uniform(-15, 15)
            for side in [-1, 1]:
                jx = cx + side * gap // 2
                pts = np.array([
                    [jx - jaw_w // 2, cy - jaw_h // 2],
                    [jx + jaw_w // 2, cy - jaw_h // 2],
                    [jx + jaw_w // 2, cy + jaw_h // 2],
                    [jx - jaw_w // 2, cy + jaw_h // 2],
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], (85, 90, 95))
                # Serration lines within jaw
                for sy in range(cy - jaw_h // 2, cy + jaw_h // 2, 5):
                    cv2.line(img, (jx - jaw_w // 2, sy), (jx + jaw_w // 2, sy), (75, 80, 85), 1)
            shape_desc = f"dual_jaw ({jaw_w*2}x{jaw_h} px, gap {gap} px)"

        elif tool in ("wrench", "vice_grips"):
            # Single jaw impression with serrations
            jaw_w = rng.randint(40, 80)
            jaw_h = rng.randint(80, 150)
            cv2.rectangle(img, (cx - jaw_w // 2, cy - jaw_h // 2),
                          (cx + jaw_w // 2, cy + jaw_h // 2), (85, 90, 95), -1)
            # Cross-hatching
            for sy in range(cy - jaw_h // 2, cy + jaw_h // 2, 4):
                cv2.line(img, (cx - jaw_w // 2, sy), (cx + jaw_w // 2, sy), (75, 80, 85), 1)
            for sx in range(cx - jaw_w // 2, cx + jaw_w // 2, 4):
                cv2.line(img, (sx, cy - jaw_h // 2), (sx, cy + jaw_h // 2), (75, 80, 85), 1)
            shape_desc = f"rectangular ({jaw_w}x{jaw_h} px)"

        img = cv2.GaussianBlur(img, (3, 3), 0.7)

        # Scale bar
        cv2.rectangle(img, (20, h - 40), (120, h - 35), (255, 255, 255), -1)
        cv2.putText(img, "5mm", (45, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        metadata = {
            "tool_type": tool,
            "material": material,
            "impression_center": [cx, cy],
            "impression_shape": shape_desc,
            "depth_estimate": random.choice(["shallow", "moderate", "deep"]),
            "force_estimate": random.choice(["light", "moderate", "heavy"]),
            "individual_characteristics": rng.randint(3, 12),
            "magnification": random.choice(["2x", "5x", "10x"]),
        }
        return img, metadata

    def _generate_cut_mark(
        self, img_id: str, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate cut marks (e.g., wire cutters, bolt cutters on metal)."""
        rng = np.random.RandomState(seed + 700)
        h, w = 512, 512
        material = random.choice(["metal", "plastic"])

        img = self._make_surface(h, w, rng, material)

        tool = random.choice(["wire_cutters", "bolt_cutters", "tin_snips", "hacksaw"])
        cut_angle = 0.0
        mark_desc = "unknown"

        if tool == "hacksaw":
            # Multiple parallel cut lines
            cut_y = h // 2 + rng.randint(-40, 40)
            cut_depth = rng.randint(80, 200)
            num_teeth = rng.randint(15, 40)
            tooth_spacing = cut_depth / num_teeth

            for t in range(num_teeth):
                y = int(cut_y - cut_depth // 2 + t * tooth_spacing)
                depth_var = rng.randint(1, 3)
                cv2.line(img, (80, y), (w - 80, y + rng.randint(-1, 1)),
                         (70 + rng.randint(0, 20),) * 3, depth_var)

            # Cut channel
            cv2.rectangle(img, (80, cut_y - 3), (w - 80, cut_y + 3), (60, 65, 70), -1)
            mark_desc = f"parallel_cuts (teeth: {num_teeth}, depth: {cut_depth}px)"

        else:
            # Single shear cut
            cut_x = w // 2 + rng.randint(-30, 30)
            cut_angle = rng.uniform(-20, 20)

            # Deformation zone
            zone_width = rng.randint(20, 50)
            for dx in range(-zone_width, zone_width):
                x = cut_x + dx
                if 0 <= x < w:
                    col_factor = 1 - abs(dx) / zone_width
                    darken = int(60 * col_factor)
                    img[:, x] = np.clip(
                        img[:, x].astype(np.int16) - darken, 0, 255
                    ).astype(np.uint8)

            # Sharp cut line
            y_offset = int(h * math.tan(math.radians(cut_angle)))
            cv2.line(img, (cut_x, 50), (cut_x + y_offset, h - 50), (40, 45, 50), 2)

            # Burr marks along cut
            for y in range(50, h - 50, rng.randint(5, 15)):
                bx = cut_x + int((y - 50) / (h - 100) * y_offset) + rng.randint(-3, 3)
                cv2.circle(img, (bx, y), rng.randint(1, 4), (120, 125, 130), -1)

            mark_desc = f"shear_cut (angle: {cut_angle:.1f}°, width: {zone_width}px)"

        img = cv2.GaussianBlur(img, (3, 3), 0.5)

        # Scale bar
        cv2.rectangle(img, (20, h - 40), (120, h - 35), (255, 255, 255), -1)
        cv2.putText(img, "3mm", (45, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        metadata = {
            "tool_type": tool,
            "material": material,
            "cut_description": mark_desc,
            "cut_angle_deg": round(cut_angle if tool != "hacksaw" else 0, 1),
            "surface_finish": random.choice(["smooth", "rough", "striated"]),
            "individual_characteristics": rng.randint(2, 10),
            "magnification": random.choice(["2x", "5x", "10x"]),
        }
        return img, metadata

    def _generate_pry_mark(
        self, img_id: str, seed: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate pry marks (e.g., from crowbar or screwdriver on door frame)."""
        rng = np.random.RandomState(seed + 800)
        h, w = 512, 512
        material = random.choice(["wood", "metal", "painted"])

        img = self._make_surface(h, w, rng, material)

        tool = random.choice(["crowbar", "screwdriver", "pry_bar", "chisel"])

        # Pry mark = wedge-shaped depression
        cx = w // 2 + rng.randint(-50, 50)
        cy = h // 2 + rng.randint(-50, 50)
        mark_width = rng.randint(15, 40)
        mark_height = rng.randint(60, 150)
        angle = rng.uniform(-30, 30)

        # Create wedge shape
        pts = np.array([
            [cx - mark_width, cy - mark_height // 2],
            [cx + mark_width, cy - mark_height // 2],
            [cx + mark_width // 3, cy + mark_height // 2],
            [cx - mark_width // 3, cy + mark_height // 2],
        ], dtype=np.float32)

        # Rotate
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        pts_rot = np.hstack([pts, np.ones((4, 1))]) @ M.T
        pts_int = pts_rot.astype(np.int32)

        # Draw depression
        cv2.fillPoly(img, [pts_int], (70, 75, 80))
        cv2.polylines(img, [pts_int], True, (55, 60, 65), 2)

        # Material displacement (raised edges)
        for _ in range(rng.randint(5, 15)):
            px = cx + rng.randint(-mark_width - 20, mark_width + 20)
            py = cy + rng.randint(-mark_height // 2 - 20, mark_height // 2 + 20)
            cv2.circle(img, (px, py), rng.randint(3, 8), (190, 195, 200), -1)

        # Striation lines within the pry mark
        for j in range(rng.randint(5, 15)):
            ly = cy - mark_height // 2 + j * mark_height // 15
            cv2.line(img, (cx - mark_width + 5, ly), (cx + mark_width - 5, ly + rng.randint(-2, 2)),
                     (60, 65, 70), 1)

        img = cv2.GaussianBlur(img, (3, 3), 0.6)

        # Scale bar
        cv2.rectangle(img, (20, h - 40), (170, h - 35), (255, 255, 255), -1)
        cv2.putText(img, "10mm", (60, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        metadata = {
            "tool_type": tool,
            "material": material,
            "pry_mark_width_px": mark_width * 2,
            "pry_mark_height_px": mark_height,
            "insertion_angle_deg": round(angle, 1),
            "depth_estimate": random.choice(["shallow", "moderate", "deep"]),
            "material_displacement": random.choice(["minimal", "moderate", "significant"]),
            "surface_damage": random.choice(["paint_chipped", "metal_gouged", "wood_splintered"]),
            "individual_characteristics": rng.randint(3, 10),
            "magnification": random.choice(["1x", "2x", "5x"]),
        }
        return img, metadata
