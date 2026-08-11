# Changelog

## Unreleased — `Krea2 Character`: people as inputs, not JSON

### New node: `Krea2 Character (By Fedor)`

One node per person, chained into V3 / V9 / V12 through a new `characters`
input. Each node carries a **real LoRA dropdown** (`lora_name`), `strength`,
`enable`, an **`ref_image` IMAGE socket**, plus `prompt`, `name`, `ref_enable`
and `portrait`.

```
[Character 1] → [Character 2] → [Krea2 Regional Multi-LoRA V3/V9/V12]
```

Chain order is box order, exactly like the region rows it replaces.

Why: `regions_json` was the only way to set a LoRA from the API, and a free-typed
name was never checked against disk — a typo or the copied doc placeholder
`character_A.safetensors` surfaced several frames later as a bare
`FileNotFoundError` from inside safetensors. A dropdown makes that class of
failure impossible. The IMAGE socket also lifts the old restriction that a
reference had to be uploaded through the node's own button: any `IMAGE` source
works now (LoadImage, a crop, an upscale).

**The chain writes `regions_json` for you.** `web/krea2_character.js` walks the
chain in the browser and populates the region rows live, so LoRA names,
strengths and prompts appear in the existing row UI the moment you wire or edit
a character — the chain is visibly the thing driving the render, not an
invisible override. Unwiring restores the text that was there before.

Reference images stay on their sockets and leave `ref_image` blank in the rows:
a wired ref is a live IMAGE tensor with no filename, so a placeholder would
break the row's thumbnail fetch and linger as a dead filename if the chain were
deleted. The badge under the node reports how many characters carry one.

At render time the server still rebuilds the rows from the chain, so API calls
and headless runs behave identically to the UI.

`characters` is declared **last** in `optional` and with **`forceInput: True`**,
for the two reasons the surrounding code already documents:

- widget order is the workflow save format, so inserting above existing widgets
  re-reads every later value in saved graphs (the bug that once put `1536` into
  `edit_lora`);
- without `forceInput` the frontend attaches a widget slot to the input, which
  serialises a `null` into both `widgets_values` and the API prompt. That null
  overrides the wired link, the chain arrives as `None`, and the node renders
  from `regions_json` — LoRAs silently absent from the result.

The browser-side chain walk uses `node.getInputNode()` rather than indexing
`app.graph.links`, which is a `Map` in frontend 1.48.x and a plain object in
older builds; indexing it directly returns `undefined` and reads as "nothing is
wired".

### Tests

`pytest tests/` — 57 tests, no GPU and no model weights required. A stub
ModelPatcher captures the armed wrapper and post-CFG hook, and a stub VAE
encodes each reference to a constant latent equal to its mean pixel value, so
"did this character's reference reach this character's box" is read straight off
the denoised tensor. LoRAs are tiny generated safetensors in a private folder
registered via `add_model_folder_path`, so the suite never touches real models.

Covered: chain building and ordering, the JSON transport (including the
double-wrap unwrap and the loud garbage fallback), the ref-tensor registry and
its bound, `_resolve_lora_path`, the INPUT_TYPES invariants (`characters` last,
`forceInput`, V12 inheriting V9), and end-to-end `apply` — LoRA scaling by
`base_strength`, box re-claiming when a character is disabled, `ref_enable`,
missing VAE, and the sampling window.

The config lives in `tests/pytest.ini` on purpose: this directory is a package,
so a rootdir above it makes pytest build a Package collector and import
`__init__.py` standalone, where its relative imports fail before any test runs.

### Fixed

- `_resolve_lora_path` no longer falls through with an unresolved name. A LoRA
  that isn't in any `models/loras` folder now raises at the node, naming the
  file and where it was looked for, instead of failing later inside
  `safetensors.torch.load_file`. Hand-typed absolute paths still work.

## 2.0.0 — V12 Unified Spatial + krea2edit + Regional Detailer

The headline: **bounding boxes now control WHERE and HOW LARGE each subject
renders**, not just where its LoRA is allowed to act — and you can drop your
LoRA characters into any real photo (scene transfer) or dress them from a
second photo (outfit transfer) **with nothing but the standard identity edit
LoRA**. No per-character reference photos, no portrait pre-renders required.

### New node: `Krea2 Regional Multi-LoRA V12 (Unified Spatial)`

- **Unified caption with exact token spans.** Instead of appending separate
  per-region text encodings, V12 compiles your box-builder prompt into ONE
  scene-wide caption and resolves the exact Qwen token span of each region's
  subject clause — including the offset math to locate the caption inside the
  grounded (vision-token-prefixed) encoding. Every routing decision operates
  on the real tokens, not approximations.
- **Fused block-sparse attention ownership.** A FlexAttention block mask
  gives each region's text span exclusive cross-modal ownership of a field
  inside its box: subject A's tokens cannot influence subject B's pixels and
  vice versa. This is a hard block, not a bias — the main cure for identity
  bleeding and "one box accurate, the neighbor generic".
- **Attraction field.** Hard blocking alone only *prevents* leakage; nothing
  pulls a subject into its box. V12 adds a pre-softmax logit boost binding
  each regional span to its full box (encoded as indicator channels on Q/K,
  which survives torch.compile where score_mod does not). Subjects
  materialize inside their boxes instead of wherever the model prefers.
- **Box-authoritative framing.** The compiled caption's camera sentence is
  derived from the largest active box height, and conflicting close-up
  wording (e.g. "selfie" with knee-high boxes) is rewritten automatically.
  Box size is the framing contract: tall box = large subject, small box =
  distant subject.
- **LoRA mask skirts with a Voronoi limit.** Delta masks extend past the box
  edge (35% of box size, 6%-of-canvas floor) so a subject that slightly
  overflows keeps full identity — but each skirt stops halfway across the
  gap to any neighboring box, so skirts can never cause cross-identity
  bleed.
- **Per-box feather cap.** Feathering is now capped per box (30% of the
  box's smaller side). Previously the canvas-fraction feather could consume
  a tiny box's whole interior, silently running its LoRA at ~half weight —
  tiny boxes now always reach a full-strength core.
- **Overlap competition.** Where two LoRA masks overlap, the region with the
  strongest inward mask wins outright (winner-take-all) instead of both
  identities summing into a blend.
- **Coexists with V9/V3/v1.** V12 is a separate node and module; existing
  workflows using the older nodes are untouched.

### Scene + outfit transfer (krea2edit path) — only the edit LoRA required

- Wire any photo into `extra_ref_1` and the whole generation is re-composed
  inside that scene: global lighting, perspective and reflections integrate
  naturally because the image is generated from noise with the scene as a
  reference frame — not latent-pasted.
- Wire a second photo into `extra_ref_2` (full-canvas box) and give it a
  role in `refs_json` (e.g. `{"role":"object","note":"outfit, worn by the
  woman"}`) for outfit/object/style transfer. The node writes the referring
  text with the correct frame number automatically.
- Requires only the standard Krea2 identity edit LoRA in the `edit_lora`
  slot. Character likeness comes entirely from your per-region character
  LoRAs — portraits and per-region reference photos are optional extras,
  not requirements.

### New node: `Krea2 Regional Detailer`

- Optional post-pass that re-renders each subject at high resolution with
  its OWN character LoRA: a body pass over each planned box, then a face
  pass anchored to **detected** faces (YOLOv8 `face_yolov8m.pt` if
  installed, OpenCV Haar fallback).
- Faces are detected on the full final image and assigned one-to-one to
  regions by proximity — so even a subject that drifted out of its box gets
  its correct LoRA applied exactly where its face actually rendered. This
  recovers likeness in the residual failure case where a subject renders
  across the box seam.
- Feathered seamless paste-back; total pixel budget capped so a huge box
  cannot exceed the main generation's cost; `skip_above_px` gate to only
  refine small/distant subjects.

### Fixes

- **Major trainer key compatibility**: the shared regional loader now
  canonicalizes Krea 2 LoRA keys from AI Toolkit/Diffusers/PEFT,
  Musubi/Kohya, and OneTrainer. Repeated wrappers such as
  `base_model.model.transformer.`, flat `lora_unet_` /
  `lora_transformer_` names, OneTrainer `__` separators, and named PEFT
  adapters are supported. Native ComfyUI and LoKr paths remain unchanged.
  Real 264-layer Diffusers fixtures now translate 264/264 modules instead of
  matching zero; native 256-layer LoRA and LoKr fixtures remain 256/256.
- **Public package import error**: `__init__.py` referenced modules that
  were not in the repository, breaking fresh installs. The registration
  list now matches the shipped files exactly.
- **Memory-leak warnings / VRAM pressure**: regional sessions now hold weak
  references to the model patcher and UNet layers, so ComfyUI can actually
  garbage-collect swapped models ("Potential memory leak" warnings gone).
- **Widget corruption via BOUNDING_BOX widgets**: all `BOUNDING_BOX` inputs
  are `forceInput` sockets, preventing the Node 2.0 frontend from injecting
  phantom x/y/width/height widgets that phase-shifted every saved value
  after them.
- **OOM at high resolution**: FlexAttention is compiled with dynamic shapes
  (no recompile-limit fallback to a dense 17 GB score tensor) and uses
  reduced kernel tiles so the fused kernel fits shared memory even with the
  attraction channels.
- **OOM with native-resolution reference frames**: block-mask construction is
  now compiled, so it reduces directly to sparse blocks instead of first
  materializing a dense Q-by-K mask. The reported 63,662-token case dropped
  from a 30.27 GiB allocation request to a measured 99 MiB peak.
- **Duplicate-subject / cardinality bugs**: centered semantic ownership
  cores and per-region cardinality phrasing stop one wide box from seeding
  two people.
- Tooltips added or expanded on nearly every input of the V9/V12 and
  Detailer nodes — hover any field for what it does and how to set it.

## 1.x

- **v3** — Reference Lock: per-region reference images via latent-mold
  guidance at the sampler (identity anchoring across generations), per-row
  reference upload with inline thumbnails, ref-only regions, scheduled
  guidance window.
- **LoKr support** — Kronecker-factored LoRAs (`lokr_w1`/`lokr_w2`) from
  newer ai-toolkit builds now load and apply correctly.
- **Box-row sync fixes** — region rows reliably track box creation and
  deletion in the builder.
- **v1** — original release: hard per-box LoRA masking via masked
  activation-delta injection.
