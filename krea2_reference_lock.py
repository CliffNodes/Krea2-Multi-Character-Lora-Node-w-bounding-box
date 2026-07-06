"""
Krea2ReferenceLock / Krea2ReferenceLockMulti (By Fedor) - v2 "latent mold"
reference steering.

Sculptor's-mold guidance: a reference image is VAE-encoded into a latent
"cast". At every sampling step (within a scheduled window) the model's
predicted-clean latent (x0 / denoised) is compared against the cast inside a
target bounding box and nudged toward it:

    denoised[box] += strength * mask * (mold - denoised[box])

Krea2 has NO native reference-latent pathway (its DiT sequence is strictly
[text | image] and extra_conds discards reference_latents), so this operates
one layer up - at the sampler's post-CFG hook - which is model-agnostic and
never touches model weights (fp8-safe). Same intervention family as latent-
anchor nodes for LTX2, adapted to Krea2 stills + explicit bboxes.

Two nodes:
  * Krea2ReferenceLock       - one reference -> one box. Chainable.
  * Krea2ReferenceLockMulti  - dynamic reference_i inputs; reference i -> box i,
    matching the region order of Krea2RegionalMultiLoRA. You SEE which image
    each box/LoRA references. (Dynamic inputs added by web/*.js.)

Tier-1 characteristics:
  * Anchors composition AND identity inside the box toward the reference.
  * Couples pose/framing to the reference; use start/end_percent to guide only
    the early-mid steps, then release the model to integrate.
  * Stacks cleanly with Krea2RegionalMultiLoRA (different intervention points).
"""

import logging
import re

import torch
import torch.nn.functional as F

from .krea2_regional_multilora import (
    _coerce_bbox_norm,
    _normalize_bboxes,
    _rect_token_mask,
)


# ---------------------------------------------------------------------------
# shared latent-mold helpers
# ---------------------------------------------------------------------------
def _latent_rect_mask(rows, cols, box, feather, device, dtype):
    """Feathered 2D mask (1,1,rows,cols) for a normalised (x0,y0,x1,y1) box."""
    x0, y0, x1, y1 = box
    m = _rect_token_mask(rows, cols, x0, y0, x1, y1, feather)
    return m.reshape(1, 1, rows, cols).to(device=device, dtype=dtype)


def _encode_reference(model_clone, vae, image):
    """IMAGE -> reference latent in the model's *processing* space."""
    pixels = image[:, :, :, :3]
    storage = vae.encode(pixels)
    return model_clone.model.process_latent_in(storage)


def _build_mold(ref_model_space, box, C, H, W, feather, device):
    """Return (mold[1,C,H,W], mask[1,1,H,W]) placing the resized reference latent
    inside the box, else None on a channel mismatch / degenerate box."""
    ref = ref_model_space
    if ref.shape[1] != C:
        logging.warning("[Krea2ReferenceLock] ref latent has %d channels, latent has %d; skipping.",
                        ref.shape[1], C)
        return None
    x0, y0, x1, y1 = box
    bx0, bx1 = int(round(x0 * W)), int(round(x1 * W))
    by0, by1 = int(round(y0 * H)), int(round(y1 * H))
    bx0 = max(0, min(bx0, W - 1))
    by0 = max(0, min(by0, H - 1))
    bx1 = min(max(bx1, bx0 + 1), W)
    by1 = min(max(by1, by0 + 1), H)
    ref = ref[:1].to(device=device, dtype=torch.float32)
    fitted = F.interpolate(ref, size=(by1 - by0, bx1 - bx0), mode="bilinear", align_corners=False)
    mold = torch.zeros(1, C, H, W, device=device, dtype=torch.float32)
    mold[:, :, by0:by1, bx0:bx1] = fitted
    mask = _latent_rect_mask(H, W, box, feather, device, torch.float32)
    return mold, mask


def _sigma_window(model_clone, start_percent, end_percent):
    ms = model_clone.get_model_object("model_sampling")
    return (ms.percent_to_sigma(float(start_percent)),
            ms.percent_to_sigma(float(end_percent)))


def _in_window(sigma, sigma_start, sigma_end):
    sv = float(sigma.max().item()) if torch.is_tensor(sigma) else float(sigma)
    # sigma runs high -> low; window is [sigma_end, sigma_start].
    return not (sv > sigma_start + 1e-9 or sv < sigma_end - 1e-9)


_COMMON_WINDOW_INPUTS = {
    "strength": ("FLOAT", {
        "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": ("Per-step pull toward the reference inside the box. "
                    "0.2-0.4 = strong anchor that still integrates; 0.7+ approaches a paste."),
    }),
    "start_percent": ("FLOAT", {
        "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": "Start of the guidance window (fraction of sampling).",
    }),
    "end_percent": ("FLOAT", {
        "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": ("End of the guidance window. Ending ~0.5-0.7 locks identity early, "
                    "then lets the model blend seams/lighting naturally."),
    }),
    "feather": ("FLOAT", {
        "default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
        "tooltip": "Soft edge of the guidance mask (fraction of latent grid).",
    }),
    "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                             "tooltip": "Canvas width used to interpret pixel-space bboxes."}),
    "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
}


# ---------------------------------------------------------------------------
# single-reference node (chainable)
# ---------------------------------------------------------------------------
class Krea2ReferenceLock:
    """Steers the in-progress denoised latent toward one VAE-encoded reference
    image inside one bounding box (post-CFG guidance; no weight changes)."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "reference_image": ("IMAGE", {"tooltip": "The 'mold': what this box converges toward."}),
        }
        req.update(_COMMON_WINDOW_INPUTS)
        return {
            "required": req,
            "optional": {
                "bboxes": ("BOUNDING_BOX", {"tooltip": "Boxes from the same builder feeding the MultiLoRA node."}),
                "box_index": ("INT", {"default": 0, "min": 0, "max": 63,
                                      "tooltip": "Which wired box this reference locks onto (0-based)."}),
                "box_x0": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Manual box (normalised) when no bboxes are wired."}),
                "box_y0": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_x1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_y1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Krea2/By Fedor"

    DESCRIPTION = (
        "Krea2 Reference Lock (By Fedor, v2 beta). Latent-mold guidance: encodes a "
        "reference image and nudges the in-progress denoised latent toward it inside "
        "a bounding box, each step, over a scheduled window. Chain one node per box."
    )

    def apply(self, model, vae, reference_image, strength, start_percent, end_percent,
              feather, canvas_width, canvas_height, bboxes=None, box_index=0,
              box_x0=0.0, box_y0=0.0, box_x1=1.0, box_y1=1.0):
        if strength <= 0.0 or end_percent <= start_percent:
            logging.info("[Krea2ReferenceLock] disabled (strength/window); passthrough.")
            return (model,)

        box = None
        frame = _normalize_bboxes(bboxes)
        if frame:
            if box_index < len(frame):
                box = _coerce_bbox_norm(frame[box_index], int(canvas_width), int(canvas_height))
            else:
                logging.warning("[Krea2ReferenceLock] box_index %d out of range (%d boxes); manual box.",
                                box_index, len(frame))
        if box is None:
            box = (min(box_x0, box_x1), min(box_y0, box_y1),
                   max(box_x0, box_x1), max(box_y0, box_y1))
        if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
            logging.warning("[Krea2ReferenceLock] degenerate box %s; passthrough.", box)
            return (model,)

        m = model.clone()
        ref_ms = _encode_reference(m, vae, reference_image)
        sigma_start, sigma_end = _sigma_window(m, start_percent, end_percent)
        w, fth = float(strength), float(feather)
        state = {"key": None, "mm": None, "logged": False}

        def post_cfg(args):
            denoised = args["denoised"]
            if denoised.dim() != 4 or not _in_window(args["sigma"], sigma_start, sigma_end):
                return denoised
            C, H, W = denoised.shape[1], denoised.shape[2], denoised.shape[3]
            if state["key"] != (C, H, W):
                state["mm"] = _build_mold(ref_ms, box, C, H, W, fth, denoised.device)
                state["key"] = (C, H, W)
                if state["mm"] is not None and not state["logged"]:
                    logging.info("[Krea2ReferenceLock] mold armed: latent %dx%d box=%s "
                                 "window sigma %.4f->%.4f strength %.2f",
                                 W, H, tuple(round(v, 3) for v in box), sigma_start, sigma_end, w)
                    state["logged"] = True
            if state["mm"] is None:
                return denoised
            mold, mask = state["mm"]
            d32 = denoised.float()
            return (d32 + (w * mask) * (mold - d32)).to(denoised.dtype)

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)


# ---------------------------------------------------------------------------
# multi-reference node (dynamic reference_i inputs; reference i -> box i)
# ---------------------------------------------------------------------------
class Krea2ReferenceLockMulti:
    """Dynamic multi-reference latent-mold guidance. Connect reference_0,
    reference_1, ... (slots grow via web JS); reference i steers box i, matching
    the region order of Krea2RegionalMultiLoRA. One node, all references."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {"model": ("MODEL",), "vae": ("VAE",)}
        req.update(_COMMON_WINDOW_INPUTS)
        return {
            "required": req,
            "optional": {
                "bboxes": ("BOUNDING_BOX", {
                    "tooltip": "Boxes from the same builder feeding the MultiLoRA node. "
                               "reference_i maps to box i.",
                }),
                # reference_0, reference_1, ... are added dynamically by web JS and
                # arrive via **kwargs (verified: ComfyUI gathers linked inputs even
                # when undeclared, and validation ignores extras).
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Krea2/By Fedor"

    DESCRIPTION = (
        "Krea2 Reference Lock — Multi (By Fedor, v2 beta). Dynamic reference_i inputs, "
        "one per bounding box, in the same order as the MultiLoRA regions. Each box's "
        "denoised latent is steered toward its reference over a scheduled window. "
        "No model-weight changes."
    )

    def apply(self, model, vae, strength, start_percent, end_percent, feather,
              canvas_width, canvas_height, bboxes=None, **kwargs):
        if strength <= 0.0 or end_percent <= start_percent:
            logging.info("[Krea2ReferenceLockMulti] disabled (strength/window); passthrough.")
            return (model,)

        # Collect reference_i tensors, keyed by their numeric index (= box index).
        refs = []
        for k, v in kwargs.items():
            m = re.match(r"^reference_(\d+)$", k)
            if m and torch.is_tensor(v):
                refs.append((int(m.group(1)), v))
        if not refs:
            logging.warning("[Krea2ReferenceLockMulti] no reference images connected; passthrough.")
            return (model,)
        refs.sort(key=lambda t: t[0])

        frame = _normalize_bboxes(bboxes)
        cw, ch = int(canvas_width), int(canvas_height)
        m = model.clone()

        entries = []  # (box_norm, ref_model_space)
        for idx, img in refs:
            if frame and idx < len(frame):
                box = _coerce_bbox_norm(frame[idx], cw, ch)
            else:
                logging.warning("[Krea2ReferenceLockMulti] reference_%d has no box "
                                "(%d boxes wired); skipping.", idx, len(frame))
                continue
            if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
                logging.warning("[Krea2ReferenceLockMulti] reference_%d degenerate box; skipping.", idx)
                continue
            entries.append((box, _encode_reference(m, vae, img)))

        if not entries:
            logging.warning("[Krea2ReferenceLockMulti] no reference mapped to a valid box; passthrough.")
            return (model,)

        sigma_start, sigma_end = _sigma_window(m, start_percent, end_percent)
        w, fth = float(strength), float(feather)
        state = {"key": None, "built": [], "logged": False}

        def post_cfg(args):
            denoised = args["denoised"]
            if denoised.dim() != 4 or not _in_window(args["sigma"], sigma_start, sigma_end):
                return denoised
            C, H, W = denoised.shape[1], denoised.shape[2], denoised.shape[3]
            if state["key"] != (C, H, W):
                built = []
                for box, ref_ms in entries:
                    mm = _build_mold(ref_ms, box, C, H, W, fth, denoised.device)
                    if mm is not None:
                        built.append(mm)
                state["built"] = built
                state["key"] = (C, H, W)
                if not state["logged"]:
                    logging.info("[Krea2ReferenceLockMulti] %d molds armed: latent %dx%d "
                                 "window sigma %.4f->%.4f strength %.2f",
                                 len(built), W, H, sigma_start, sigma_end, w)
                    state["logged"] = True
            if not state["built"]:
                return denoised
            d32 = denoised.float()
            for mold, mask in state["built"]:
                d32 = d32 + (w * mask) * (mold - d32)
            return d32.to(denoised.dtype)

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "Krea2ReferenceLock": Krea2ReferenceLock,
    "Krea2ReferenceLockMulti": Krea2ReferenceLockMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2ReferenceLock": "Krea2 Reference Lock — Latent Mold (By Fedor)",
    "Krea2ReferenceLockMulti": "Krea2 Reference Lock — Multi (By Fedor)",
}
