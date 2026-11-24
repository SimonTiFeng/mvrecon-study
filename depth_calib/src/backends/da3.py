"""DA3 Mono / DA3 Metric adapters (frozen checkpoints, no training).

Uses the vendored official repo at the pinned commit. The export package is
shimmed because it eagerly imports heavy optional dependencies (moviepy,
pycolmap) that this project never uses; export_dir is always None here.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import types

import cv2
import numpy as np
import torch
from PIL import Image

_ROOTS = pathlib.Path(__file__).resolve().parents[2]
_VENDOR = _ROOTS / "vendor" / "Depth-Anything-3" / "src"
_VENDOR_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

_api_cls = None


def _api():
    global _api_cls
    if _api_cls is not None:
        return _api_cls
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    if "depth_anything_3.utils.export" not in sys.modules:
        shim = types.ModuleType("depth_anything_3.utils.export")

        def _disabled(*_a, **_k):
            raise RuntimeError("DA3 export is disabled in this project")

        shim.export = _disabled
        sys.modules["depth_anything_3.utils.export"] = shim
    from depth_anything_3.api import DepthAnything3
    _api_cls = DepthAnything3
    return _api_cls


class DA3Backend:
    def __init__(self, spec: dict, inference_cfg: dict):
        self.spec = spec
        self.cfg = inference_cfg
        self.model = None
        self.checkpoint_sha256 = None
        self.precision_note = None

    def _patch_autocast_for_precision(self) -> None:
        """Force fp16 autocast on pre-SM80 GPUs when the config asks for it.

        DA3 (vendor api.py:126) selects bf16 whenever torch.cuda.is_bf16_supported()
        is true. On V100 (sm_70) that path is emulated, so a run can succeed while
        computing in an unintended precision. The policy is declared in YAML
        (`prefer_fp16_on_pre_sm80`, `min_sm80_for_bf16`), never hardcoded here.
        """
        if not self.cfg.get("prefer_fp16_on_pre_sm80", False):
            self.precision_note = "vendor default autocast"
            return
        if not torch.cuda.is_available():
            self.precision_note = "cpu (no autocast policy applied)"
            return
        major, _ = torch.cuda.get_device_capability(0)
        threshold = (int(self.cfg.get("min_sm80_for_bf16", 80)) // 10)
        if major >= threshold:
            self.precision_note = f"vendor default autocast (sm_{major}0 >= threshold)"
            return
        import depth_anything_3.api as da3_api

        if getattr(da3_api, "_depthcalib_fp16_patch", False):
            self.precision_note = f"fp16 autocast forced for sm_{major}0 (already patched)"
            return
        orig_forward = da3_api.DepthAnything3.forward

        def forward(self_, image, *a, **kw):
            with torch.no_grad():
                with torch.autocast(device_type=image.device.type, dtype=torch.float16):
                    return self_.model(image, *a, **kw)

        da3_api.DepthAnything3.forward = forward
        da3_api._depthcalib_fp16_patch = True
        self.precision_note = f"fp16 autocast forced for sm_{major}0"

    def load(self) -> "DA3Backend":
        api = _api()
        target = self.spec.get("local_dir") or self.spec["hf_repo"]
        if self.spec.get("local_dir"):
            ld = pathlib.Path(self.spec["local_dir"])
            for f in ("config.json", self.spec["weight_file"]):
                if not (ld / f).exists():
                    raise FileNotFoundError(f"missing {f} in local model dir {ld}")
        self.model = api.from_pretrained(target, revision=self.spec["revision"])
        device = self.cfg.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        self._patch_autocast_for_precision()
        self.checkpoint_sha256 = _checkpoint_sha(self.spec)
        return self

    @torch.no_grad()
    def predict(self, image_path: str, original_size: tuple[int, int]) -> dict:
        """Run inference; returns raw depth resampled to (W,H) plus provenance."""
        img = Image.open(image_path).convert("RGB")
        pred = self.model.inference(
            [img], process_res=self.cfg["process_res"],
            process_res_method=self.cfg["process_res_method"])
        depth = np.asarray(pred.depth)[0].astype(np.float32)
        conf = None if pred.conf is None else np.asarray(pred.conf)[0].astype(np.float32)
        W, H = original_size
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            if conf is not None:
                conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_LINEAR)
        return {
            "depth": depth,
            "conf": conf,
            "is_metric": int(getattr(pred, "is_metric", 0) or 0),
            "processed_shape": list(map(int, np.shape(np.asarray(pred.depth)[0]))),
            "model_revision": self.spec["revision"],
            "vendor_commit": _VENDOR_COMMIT,
            "checkpoint_sha256": self.checkpoint_sha256,
            "process_res": self.cfg["process_res"],
            "process_res_method": self.cfg["process_res_method"],
            "precision_policy": self.precision_note,
        }


def _checkpoint_sha(spec: dict) -> str | None:
    try:
        if spec.get("local_dir"):
            p = pathlib.Path(spec["local_dir"]) / spec["weight_file"]
        else:
            from huggingface_hub import hf_hub_download
            p = pathlib.Path(hf_hub_download(spec["hf_repo"], spec["weight_file"],
                                             revision=spec["revision"]))
    except Exception:
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_factor(spec: dict, K: np.ndarray, processed_shape: tuple[int, int],
                  original_size: tuple[int, int], factor: float = 300.0) -> float:
    """focal/300 at the processed resolution (handoff 7 / pitfall 13).

    Applied exactly once by the adapter for the standalone metric checkpoint,
    which (unlike the nested model) does not apply it internally.
    """
    Hp, Wp = processed_shape
    W, H = original_size
    fx = K[0, 0] * (Wp / W)
    fy = K[1, 1] * (Hp / H)
    return float(((fx + fy) / 2.0) / factor)
