"""Unified forensic dataset downloader.

Downloads publicly available forensic datasets:
- CEDAR Signatures (document forensics - signature verification) [no auth]
- AZH Wound Classification (wound analysis) [no auth]
- Mendeley Autopsy Reports (post-mortem text) [no auth]
- Multi-LexSum Depositions (witness depositions from civil rights cases) [no auth]
- SOCOFing Fingerprints (fingerprint classification) [Kaggle auth]

All downloads use httpx and are fully automatic.
"""

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


def _require_httpx() -> Any:
    """Raise if httpx is not installed."""
    if httpx is None:
        raise ImportError("httpx is required for downloads. Install with: pip install httpx")
    return httpx


# ---------------------------------------------------------------------------
# Dataset source URLs
# ---------------------------------------------------------------------------

_CEDAR_ZIP_URL = (
    "https://github.com/nikostsagk/signature-verification"
    "/releases/download/cedar/cedar_dataset.zip"
)

_AZH_ZIP_URL = (
    "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations"
    "/archive/refs/heads/master.zip"
)

_MENDELEY_API_URL = "https://data.mendeley.com/api/datasets/n9z3v2k8wv"

_MULTILEXSUM_SOURCES_URL = (
    "https://huggingface.co/datasets/allenai/multi_lexsum"
    "/resolve/main/releases/v20220616/sources.json"
)
_MULTILEXSUM_TRAIN_URL = (
    "https://huggingface.co/datasets/allenai/multi_lexsum"
    "/resolve/main/releases/v20220616/train.json"
)

_KAGGLE_SOCOFING_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/ruizgara/socofing"
)


class ForensicDataDownloader:
    """Download and organise publicly available forensic datasets.

    Parameters
    ----------
    data_dir : Path
        Root data directory (e.g. ``<project>/data``).  Sub-directories for
        each dataset are created automatically.
    """

    def __init__(
        self,
        data_dir: Path,
        kaggle_username: str = "",
        kaggle_key: str = "",
    ) -> None:
        if httpx is None:
            raise ImportError(
                "httpx is required for ForensicDataDownloader. "
                "Install with: pip install httpx"
            )
        self._data_dir = data_dir
        self._kaggle_username = kaggle_username
        self._kaggle_key = kaggle_key

        # Target directories
        self._cedar_dir = data_dir / "images" / "document_forensics" / "cedar"
        self._azh_dir = data_dir / "images" / "wounds" / "azh"
        self._socofing_dir = data_dir / "images" / "fingerprints" / "socofing"
        self._postmortem_dir = data_dir / "postmortem" / "real"
        self._depositions_dir = data_dir / "depositions" / "real"

        # Ensure directories exist
        for d in [
            self._cedar_dir,
            self._azh_dir,
            self._socofing_dir,
            self._postmortem_dir,
            self._depositions_dir,
            data_dir / "images" / "ballistics",
            data_dir / "images" / "tool_marks",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public orchestration
    # ------------------------------------------------------------------

    async def download_all(
        self, limit_per_type: int = 20
    ) -> Dict[str, Any]:
        """Download all available forensic datasets.

        Each source is attempted independently -- a failure in one does not
        prevent the others from completing.

        Returns a summary dict with counts and any errors.
        """
        results: Dict[str, Any] = {}

        downloaders = [
            ("cedar_signatures", self.download_cedar_signatures, False),
            ("azh_wounds", self.download_azh_wounds, False),
            ("socofing_fingerprints", self.download_socofing_fingerprints, False),
            ("mendeley_autopsies", self.download_mendeley_autopsies, True),
            ("multilexsum_depositions", self.download_multilexsum_depositions, True),
        ]

        for name, method, accepts_limit in downloaders:
            try:
                if accepts_limit:
                    count = await method(limit=limit_per_type)
                else:
                    count = await method()
                results[name] = {"status": "ok", "count": count}
                logger.info(f"[{name}] downloaded {count} items")
            except Exception as exc:
                results[name] = {"status": "error", "error": str(exc)}
                logger.error(f"[{name}] failed: {exc}")

        return results

    # ------------------------------------------------------------------
    # 1. CEDAR Signatures
    # ------------------------------------------------------------------

    async def download_cedar_signatures(self) -> int:
        """Download CEDAR signature dataset (genuine + forged PNG images).

        Source: GitHub release ZIP (~2,640 images).
        Extracts to ``data/images/document_forensics/cedar/``.

        Returns the number of image files extracted.
        """
        marker = self._cedar_dir / ".downloaded"
        existing = list(self._cedar_dir.rglob("*.png"))
        if marker.exists() and len(existing) > 0:
            logger.info(
                f"CEDAR signatures already present ({len(existing)} images) -- skipping"
            )
            return len(existing)

        logger.info("Downloading CEDAR signature dataset from GitHub ...")

        _http = _require_httpx()
        async with _http.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(_CEDAR_ZIP_URL)
            resp.raise_for_status()

        logger.info(
            f"Downloaded {len(resp.content) / 1024 / 1024:.1f} MB -- extracting ..."
        )

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            image_count = 0
            for info in zf.infolist():
                if info.is_dir():
                    continue
                lower = info.filename.lower()
                if not (lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg")):
                    continue

                # Flatten or preserve relative structure
                rel = Path(info.filename)
                # Strip top-level archive folder if present
                parts = rel.parts
                if len(parts) > 1:
                    dest = self._cedar_dir / Path(*parts[1:])
                else:
                    dest = self._cedar_dir / rel.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info))
                image_count += 1

        marker.write_text("ok")
        logger.info(f"CEDAR: extracted {image_count} signature images")
        return image_count

    # ------------------------------------------------------------------
    # 2. AZH Wound Classification
    # ------------------------------------------------------------------

    async def download_azh_wounds(self) -> int:
        """Download AZH wound classification images.

        Source: GitHub repository ZIP containing nested Train.zip and Test.zip
        with wound images. Extracts to ``data/images/wounds/azh/``.

        Returns the number of image files extracted.
        """
        marker = self._azh_dir / ".downloaded"
        existing = list(self._azh_dir.rglob("*.jpg")) + list(
            self._azh_dir.rglob("*.jpeg")
        ) + list(self._azh_dir.rglob("*.png"))
        if marker.exists() and len(existing) > 0:
            logger.info(
                f"AZH wound images already present ({len(existing)} images) -- skipping"
            )
            return len(existing)

        logger.info("Downloading AZH wound classification dataset from GitHub ...")

        _http = _require_httpx()
        async with _http.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            resp = await client.get(_AZH_ZIP_URL)
            resp.raise_for_status()

        logger.info(
            f"Downloaded {len(resp.content) / 1024 / 1024:.1f} MB -- extracting images ..."
        )

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_count = 0

        with zipfile.ZipFile(io.BytesIO(resp.content)) as outer_zf:
            # The repo ZIP contains nested ZIPs (Train.zip, Test.zip)
            for info in outer_zf.infolist():
                if info.is_dir():
                    continue

                if info.filename.lower().endswith(".zip"):
                    # Extract inner ZIP
                    inner_name = Path(info.filename).stem  # "Train" or "Test"
                    logger.info(f"  extracting inner archive: {info.filename}")
                    inner_bytes = outer_zf.read(info)
                    try:
                        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
                            for inner_info in inner_zf.infolist():
                                if inner_info.is_dir():
                                    continue
                                ext = Path(inner_info.filename).suffix.lower()
                                if ext not in image_extensions:
                                    continue
                                rel = Path(inner_info.filename)
                                parts = rel.parts
                                # Preserve category structure under train/test
                                if len(parts) > 1:
                                    dest = self._azh_dir / inner_name / Path(*parts[1:])
                                else:
                                    dest = self._azh_dir / inner_name / rel.name
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(inner_zf.read(inner_info))
                                image_count += 1
                    except zipfile.BadZipFile:
                        logger.warning(f"  {info.filename} is not a valid ZIP")

                else:
                    # Direct image file in outer ZIP
                    ext = Path(info.filename).suffix.lower()
                    if ext in image_extensions:
                        dest = self._azh_dir / Path(info.filename).name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(outer_zf.read(info))
                        image_count += 1

        marker.write_text("ok")
        logger.info(f"AZH: extracted {image_count} wound images")
        return image_count

    # ------------------------------------------------------------------
    # 3. SOCOFing Fingerprints (Kaggle)
    # ------------------------------------------------------------------

    async def download_socofing_fingerprints(self) -> int:
        """Download SOCOFing fingerprint dataset from Kaggle.

        Dataset: ruizgara/socofing (~6,000 fingerprint images across
        Real, Altered-Easy, Altered-Medium, Altered-Hard categories).
        Requires Kaggle username + API key (classic format from kaggle.json).
        Extracts to ``data/images/fingerprints/socofing/``.

        Returns the number of image files extracted.
        """
        marker = self._socofing_dir / ".downloaded"
        existing = list(self._socofing_dir.rglob("*.BMP")) + list(
            self._socofing_dir.rglob("*.bmp")
        ) + list(self._socofing_dir.rglob("*.png")) + list(
            self._socofing_dir.rglob("*.jpg")
        )
        if marker.exists() and len(existing) > 0:
            logger.info(
                f"SOCOFing fingerprints already present ({len(existing)} images) -- skipping"
            )
            return len(existing)

        if not self._kaggle_username or not self._kaggle_key:
            raise ValueError(
                "Kaggle credentials required for SOCOFing dataset. "
                "Set KAGGLE_USERNAME and KAGGLE_KEY in .env"
            )

        logger.info("Downloading SOCOFing fingerprint dataset from Kaggle (~773 MB) ...")

        creds = base64.b64encode(
            f"{self._kaggle_username}:{self._kaggle_key}".encode()
        ).decode()
        headers = {"Authorization": f"Basic {creds}"}

        _http = _require_httpx()
        async with _http.AsyncClient(
            timeout=_http.Timeout(600.0, connect=30.0),
            follow_redirects=True,
        ) as client:
            resp = await client.get(_KAGGLE_SOCOFING_URL, headers=headers)
            if resp.status_code == 403:
                raise RuntimeError(
                    "Kaggle API returned 403. Ensure KAGGLE_KEY is the classic "
                    "API key from kaggle.json (not a KGAT_ token)."
                )
            resp.raise_for_status()

        logger.info(
            f"Downloaded {len(resp.content) / 1024 / 1024:.1f} MB -- extracting ..."
        )

        image_extensions = {".bmp", ".png", ".jpg", ".jpeg"}
        image_count = 0

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = Path(info.filename).suffix.lower()
                if ext not in image_extensions:
                    continue

                rel = Path(info.filename)
                parts = rel.parts
                if len(parts) > 1:
                    dest = self._socofing_dir / Path(*parts[1:])
                else:
                    dest = self._socofing_dir / rel.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info))
                image_count += 1

        marker.write_text("ok")
        logger.info(f"SOCOFing: extracted {image_count} fingerprint images")
        return image_count

    # ------------------------------------------------------------------
    # 4. Mendeley Autopsy Reports
    # ------------------------------------------------------------------

    async def download_mendeley_autopsies(self, limit: int = 20) -> int:
        """Download autopsy / post-mortem report data from Mendeley.

        Uses the Mendeley public dataset API to discover downloadable files
        and saves text content to ``data/postmortem/real/``.

        Returns the number of files saved.
        """
        marker = self._postmortem_dir / ".downloaded"
        existing = list(self._postmortem_dir.glob("*.*"))
        existing = [f for f in existing if not f.name.startswith(".")]
        if marker.exists() and len(existing) > 0:
            logger.info(
                f"Mendeley autopsy data already present ({len(existing)} files) -- skipping"
            )
            return len(existing)

        logger.info("Fetching Mendeley dataset metadata ...")

        saved = 0

        _http = _require_httpx()
        async with _http.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            # Step 1: fetch dataset metadata to discover file IDs
            try:
                meta_resp = await client.get(_MENDELEY_API_URL)
                meta_resp.raise_for_status()
                meta = meta_resp.json()
            except Exception as exc:
                logger.error(f"Mendeley API metadata fetch failed: {exc}")
                raise

            # The API returns a list of files under "files" or within versions
            files: List[Dict[str, Any]] = []
            if "files" in meta:
                files = meta["files"]
            elif "versions" in meta and meta["versions"]:
                # Try the latest version
                latest = meta["versions"][0]
                version_id = latest.get("version") or latest.get("id")
                version_url = f"{_MENDELEY_API_URL}/files?version={version_id}"
                try:
                    vr = await client.get(version_url)
                    if vr.status_code == 200:
                        files = vr.json() if isinstance(vr.json(), list) else []
                except Exception:
                    pass

            if not files:
                # Fallback: try fetching files endpoint directly
                files_url = f"{_MENDELEY_API_URL}/files"
                try:
                    fr = await client.get(files_url)
                    if fr.status_code == 200:
                        body = fr.json()
                        files = body if isinstance(body, list) else body.get("files", [])
                except Exception:
                    pass

            if not files:
                logger.warning(
                    "No files discovered in Mendeley dataset metadata. "
                    "The dataset API structure may have changed."
                )
                # Save the metadata itself as a reference
                meta_path = self._postmortem_dir / "mendeley_dataset_meta.json"
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                marker.write_text("meta_only")
                return 1

            logger.info(f"Mendeley: found {len(files)} files, downloading up to {limit} ...")

            for file_info in files[:limit]:
                file_id = file_info.get("id", "")
                filename = file_info.get("filename", file_info.get("name", f"{file_id}.dat"))
                download_url = (
                    file_info.get("download_url")
                    or file_info.get("content_details", {}).get("download_url")
                    or f"https://data.mendeley.com/public-files/datasets/n9z3v2k8wv/files/{file_id}/file_downloaded"
                )

                try:
                    dl_resp = await client.get(download_url)
                    if dl_resp.status_code != 200:
                        logger.warning(
                            f"Mendeley file {filename}: HTTP {dl_resp.status_code}"
                        )
                        continue
                except Exception as exc:
                    logger.warning(f"Mendeley file {filename} download failed: {exc}")
                    continue

                safe_name = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in filename
                )
                dest = self._postmortem_dir / safe_name
                dest.write_bytes(dl_resp.content)
                saved += 1
                logger.debug(f"  saved {safe_name} ({len(dl_resp.content)} bytes)")

        if saved > 0:
            marker.write_text("ok")
        logger.info(f"Mendeley: saved {saved} autopsy files")
        return saved

    # ------------------------------------------------------------------
    # 5. Multi-LexSum Depositions
    # ------------------------------------------------------------------

    async def download_multilexsum_depositions(self, limit: int = 20) -> int:
        """Download witness depositions / legal summaries from Multi-LexSum.

        Downloads train.json and sources.json directly from the HuggingFace
        repository (the datasets-server API does not serve this dataset).
        Dataset: allenai/multi_lexsum (civil rights litigation documents).
        Saves text files to ``data/depositions/real/``.

        Returns the number of deposition files saved.
        """
        marker = self._depositions_dir / ".downloaded"
        existing = [
            f
            for f in self._depositions_dir.glob("*.txt")
            if not f.name.startswith(".")
        ]
        if marker.exists() and len(existing) > 0:
            logger.info(
                f"Multi-LexSum depositions already present ({len(existing)} files) -- skipping"
            )
            return len(existing)

        logger.info("Downloading Multi-LexSum depositions from HuggingFace repo ...")

        saved = 0

        _http = _require_httpx()
        async with _http.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            # Download sources.json (contains case_id -> source document text mapping)
            try:
                resp = await client.get(_MULTILEXSUM_SOURCES_URL)
                resp.raise_for_status()
                sources_data = resp.json()
            except Exception as exc:
                logger.error(f"Failed to download Multi-LexSum sources.json: {exc}")
                # Fallback: download train.json which has summaries
                try:
                    resp2 = await client.get(_MULTILEXSUM_TRAIN_URL)
                    resp2.raise_for_status()
                    sources_data = None
                    train_data = resp2.json()
                except Exception as exc2:
                    logger.error(f"Failed to download train.json too: {exc2}")
                    return 0
            else:
                train_data = None

            if sources_data and isinstance(sources_data, dict):
                # sources.json: {doc_id: {doc_id, doc_text, doc_title, doc_type, ...}}
                logger.info(f"Multi-LexSum: found {len(sources_data)} documents in sources.json")
                for doc_id, doc_info in list(sources_data.items()):
                    if saved >= limit:
                        break
                    if not isinstance(doc_info, dict):
                        continue
                    text = doc_info.get("doc_text", "")
                    if not text or len(text) < 200:
                        continue
                    safe_id = "".join(
                        c if c.isalnum() or c in "._-" else "_"
                        for c in str(doc_id)
                    )
                    fname = f"{safe_id}.txt"
                    dest = self._depositions_dir / fname
                    dest.write_text(text, encoding="utf-8")
                    saved += 1

            elif train_data and isinstance(train_data, list):
                # train.json is a list of cases with summaries
                logger.info(f"Multi-LexSum: found {len(train_data)} cases in train.json")
                for case in train_data[:limit * 2]:
                    if saved >= limit:
                        break
                    case_id = case.get("id", f"MLXS-{saved:04d}")
                    summary = (
                        case.get("summary/long", "")
                        or case.get("summary_long", "")
                        or case.get("summary/short", "")
                        or ""
                    )
                    if len(summary) < 200:
                        continue
                    safe_id = "".join(
                        c if c.isalnum() or c in "._-" else "_"
                        for c in str(case_id)
                    )
                    dest = self._depositions_dir / f"{safe_id}_summary.txt"
                    dest.write_text(summary, encoding="utf-8")
                    saved += 1

        if saved > 0:
            marker.write_text("ok")
        logger.info(f"Multi-LexSum: saved {saved} deposition files")
        return saved

    # ------------------------------------------------------------------
    # Status / listing
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return counts of downloaded items per dataset type."""

        def _count(directory: Path, patterns: List[str]) -> int:
            total = 0
            for pat in patterns:
                total += len(list(directory.rglob(pat)))
            return total

        return {
            "cedar_signatures": _count(self._cedar_dir, ["*.png", "*.jpg"]),
            "azh_wounds": _count(self._azh_dir, ["*.jpg", "*.jpeg", "*.png"]),
            "socofing_fingerprints": _count(
                self._socofing_dir, ["*.BMP", "*.bmp", "*.png", "*.jpg"]
            ),
            "mendeley_autopsies": len([
                f for f in self._postmortem_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]) if self._postmortem_dir.exists() else 0,
            "multilexsum_depositions": len([
                f for f in self._depositions_dir.glob("*.txt")
                if not f.name.startswith(".")
            ]) if self._depositions_dir.exists() else 0,
        }
