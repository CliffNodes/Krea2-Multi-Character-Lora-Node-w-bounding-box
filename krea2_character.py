"""
Krea2Character (By Fedor) - one node per person, wired instead of hand-typed.

The regional nodes (V3/V9/V12) all describe their people as rows of a
``regions_json`` string. That string is the only place a LoRA filename could be
set from the API, so a typo or a copied doc placeholder (the classic
"character_A.safetensors") only surfaced deep inside safetensors at sample time.

This node makes a person an INPUT instead: a real LoRA dropdown, a strength, and
an IMAGE socket for the reference photo, so the ref can come from LoadImage, a
crop, an upscale - anything - rather than only a file uploaded through the
node's own button. Chain one node per person and wire the last one's
``characters`` output into the regional node:

    [Character 1] -> [Character 2] -> [Krea2 Regional Multi-LoRA V3/V9/V12]

A wired chain fully replaces regions_json; the text widget is left alone so
unwiring restores whatever was typed there.

Transport note: the regional nodes parse regions from JSON in three separate
places, and an IMAGE tensor cannot live in JSON. Each wired reference is parked
in ``_REF_TENSORS`` under a sentinel filename, which the shared
``_load_ref_image_tensor`` resolves back to the tensor. That keeps every
existing parse/has-ref/serialize path in V3, V9 and V12 working untouched.
"""

import itertools
import json
import logging

import folder_paths

_REF_SENTINEL = "krea2:ref:"
_REF_TENSORS = {}
_REF_IDS = itertools.count()
# Only the refs of the prompt being executed matter; older entries are dead
# weight holding image tensors alive, so keep the map small and FIFO-evicted.
_REF_KEEP = 256


def register_ref_tensor(image):
    """Park a wired IMAGE and return the sentinel filename that stands in for it."""
    key = f"{_REF_SENTINEL}{next(_REF_IDS)}"
    _REF_TENSORS[key] = image
    while len(_REF_TENSORS) > _REF_KEEP:
        _REF_TENSORS.pop(next(iter(_REF_TENSORS)))
    return key


def lookup_ref_tensor(name):
    """The tensor behind a sentinel filename, or None for a real input-folder file."""
    if isinstance(name, str) and name.startswith(_REF_SENTINEL):
        tensor = _REF_TENSORS.get(name)
        if tensor is None:
            logging.warning("[Krea2Character] wired reference %s expired; "
                            "re-run the character node.", name)
        return tensor
    return None


def characters_to_regions_json(characters, fallback_json):
    """Wired character chain -> the regions_json string every regional node parses.

    Returns ``fallback_json`` untouched when nothing is wired, so the in-node
    region rows keep working exactly as before.
    """
    if not characters:
        return fallback_json

    # ComfyUI hands a node output over as a one-element list, and some frontends
    # re-wrap it again. Peel until the chain itself is in hand, otherwise every
    # entry fails the isinstance check below and the wired chain is silently
    # ignored in favour of regions_json - the exact failure this node exists to
    # prevent.
    while (isinstance(characters, (list, tuple)) and len(characters) == 1
           and isinstance(characters[0], (list, tuple))):
        characters = characters[0]
    if isinstance(characters, dict):
        characters = [characters]

    rows = []
    for i, character in enumerate(characters):
        if not isinstance(character, dict):
            logging.warning("[Krea2Character] chain entry %d is %s, not a character; "
                            "skipped.", i, type(character).__name__)
            continue
        row = {k: v for k, v in character.items() if k != "_ref_tensor"}
        tensor = character.get("_ref_tensor")
        row["ref_image"] = register_ref_tensor(tensor) if tensor is not None else ""
        row["name"] = str(row.get("name") or f"character{i + 1}")
        rows.append(row)

    if not rows:
        # Never fail over to regions_json without saying so: a silent fallback
        # renders with the wrong (or no) LoRAs and looks like the chain "just
        # didn't work".
        logging.error("[Krea2Character] a characters chain is wired but yielded no "
                      "usable rows (got %r); falling back to regions_json.",
                      characters)
        return fallback_json

    logging.info("[Krea2Character] %d wired character(s) override regions_json: %s",
                 len(rows), ", ".join(
                     f"{r['name']}={r['lora']}@{r['strength']:.2f}"
                     f"{' +ref' if r['ref_image'] else ''}" for r in rows))
    return json.dumps(rows, indent=2)


class Krea2Character:
    """One person: LoRA + strength + optional reference image, as sockets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"), {
                    "default": "None",
                    "tooltip": (
                        "Character LoRA, picked from models/loras. Because this is a "
                        "real dropdown a missing file is impossible - unlike a name "
                        "typed into regions_json. 'None' = reference image only."
                    ),
                }),
                "strength": ("FLOAT", {
                    "default": 1.1, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": (
                        "This character's LoRA weight inside its own box. The regional "
                        "node's base_strength multiplies it."
                    ),
                }),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Off = this character claims no box, and everyone after it in "
                        "the chain shifts up one box."
                    ),
                }),
            },
            "optional": {
                "characters": ("KREA2_CHARACTERS", {
                    "tooltip": (
                        "Chain in the previous character. Chain order is box order: "
                        "the first character gets box 1, the second box 2, and so on."
                    ),
                }),
                "ref_image": ("IMAGE", {
                    "tooltip": (
                        "Reference photo for this character - any IMAGE source "
                        "(LoadImage, a crop, an upscale). The regional node molds this "
                        "character's box toward it. Needs a VAE wired on that node."
                    ),
                }),
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": (
                        "This character's description, e.g. 'a man in a grey suit'. "
                        "Used by V9/V12 for per-region text routing and portraits; "
                        "ignored by V3."
                    ),
                }),
                "name": ("STRING", {
                    "default": "",
                    "tooltip": "Label used in masks and logs. Blank = character1, character2, ...",
                }),
                "ref_enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Off = keep the reference image wired but stop it steering the "
                        "render. Handy for A/B-ing likeness without unplugging."
                    ),
                }),
                "portrait": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "V9 only: render a solo test portrait of this character. Costs "
                        "an extra pass plus a model reload - leave off once the LoRA "
                        "is known good."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("KREA2_CHARACTERS",)
    RETURN_NAMES = ("characters",)
    FUNCTION = "build"
    CATEGORY = "Fedor Nodes"

    DESCRIPTION = (
        "Krea2 Character (By Fedor). Defines one person - LoRA, strength and an "
        "optional reference IMAGE - as wired inputs instead of a row of "
        "regions_json. Chain one per person and feed the last node's 'characters' "
        "output into a Krea2 Regional Multi-LoRA V3/V9/V12 node; the chain replaces "
        "that node's regions_json entirely. Chain order is box order."
    )

    def build(self, lora_name, strength, enable, characters=None, ref_image=None,
              prompt="", name="", ref_enable=True, portrait=False):
        chain = list(characters) if characters else []
        chain.append({
            "name": str(name or "").strip() or f"character{len(chain) + 1}",
            "lora": str(lora_name or "None"),
            "strength": float(strength),
            "enable": bool(enable),
            "prompt": str(prompt or "").strip(),
            "ref_enable": bool(ref_enable),
            "portrait": bool(portrait),
            "_ref_tensor": ref_image,
        })
        return (chain,)


NODE_CLASS_MAPPINGS = {
    "Krea2Character": Krea2Character,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2Character": "Krea2 Character (By Fedor)",
}
