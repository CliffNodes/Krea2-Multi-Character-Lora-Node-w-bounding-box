"""
ComfyUI-Krea2-Regional-MultiLoRA (By Fedor)

Regional multi-LoRA for Krea 2 via masked activation-delta injection. Each
region's LoRA is constrained to its bounding box at forward time - a hard
spatial mask, not an attention-bias nudge.
"""

from .krea2_regional_multilora import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
