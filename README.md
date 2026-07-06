# ComfyUI Krea2 Regional Multi-LoRA — By Fedor

**Put multiple character LoRAs in one image — each one locked to its own box. Plus full bounding-box layout control for objects and any non-LoRA element.**

Draw a box for each character, assign a LoRA to each box, and this node guarantees that LoRA A only affects region A and LoRA B only affects region B. No bleed, no merged faces, no averaging. Works with two characters, three, four — as many as you want to draw boxes for.

And it isn't just for LoRAs: paired with the Ideogram 4-style prompt builder, you get **per-box control over everything else in the scene too** — objects, props, backgrounds, secondary subjects. Krea 2 responds to those positional box prompts the same way Ideogram 4 does, so you can lay out an entire composition by drawing and describing boxes, then drop character LoRAs into whichever boxes need a specific identity.

![Multi-character regional LoRA example workflow](assets/example_multi_character.png)

---

## The Problem

If you've tried loading two character LoRAs at once with a normal LoRA loader, you already know what happens: the two identities smear into each other. You ask for "Alice on the left, Bob on the right" and you get one person who's a 50/50 blend of both, in both spots.

That's because a normal LoRA applies **everywhere, uniformly**. The model has no instruction to keep Alice's weights on the left. Soft tricks — attention bias, prompt engineering, CFG tweaks — reduce the smearing but never fully stop it, because the model is still *allowed* to route either LoRA anywhere.

This node removes the permission entirely.

---

## How It Works

Every LoRA is, mathematically, a small correction added to the model's internal activations. For an input `x` at a given layer, the LoRA computes:

```
delta = (x @ down.T @ up.T) * scale
output = output + delta
```

Normally that `delta` is added to **every token** — every patch of the image, everywhere. This node intercepts that step and multiplies the delta by a **spatial mask** before it's added back:

```
output = output + mask * delta
```

The mask is built from the bounding box you drew. Inside the box it's `1`; outside it's `0`. So for any image token that lands outside the box:

```
output = output + (0 * delta) = output   ← LoRA does nothing
```

There is no mathematical path for the LoRA to affect anything outside its region. It's not discouraged from leaking — it is structurally incapable of it.

**Key details:**

- The mask is built at generation time from the **actual latent token grid**, so it scales correctly to whatever resolution you're generating at.
- Text tokens are always skipped (they have no spatial position) — the mask only applies to image tokens.
- LoRA matrices are loaded raw and matched to the live model layers by name, so it works on **fp8 / quantized Krea 2 checkpoints** without touching the quantized weights.
- Multiple LoRAs are injected in the same forward pass, each with its own mask. They run in parallel, not chained.
- `seam_feather` applies a soft sigmoid edge to the box boundary so you don't get a hard pixel-cut seam between regions.

---

## Features

- **Unlimited regions.** Two characters or ten — add a row per character. No code changes, no fixed slots.
- **Bounding-box control for non-LoRA elements too.** Objects, props, backgrounds, extra people — anything you can describe. Draw a box, describe it, and Krea 2 places it there, exactly like Ideogram 4's bounding-box prompting. LoRAs and plain object boxes coexist in the same layout.
- **Auto-syncing rows.** Wire a bounding-box builder into the node and the region rows appear and disappear automatically as you draw or delete boxes. Your LoRA picks are preserved when the count changes.
- **Hard spatial masking.** Activation-delta injection, not attention bias. LoRAs cannot cross their box boundary.
- **fp8-safe.** Never modifies quantized model weights; injects at forward time.
- **CLIP passes through untouched.** The regional effect is UNet-side, exactly where identity lives.

## Full Layout Control — Not Just LoRAs

The bounding boxes drive two independent things at once:

1. **Layout / content (every box).** Each box you draw in the prompt builder carries a description, and those descriptions become a positional prompt fed to Krea 2's Qwen3-VL text encoder — the same natural, on-distribution format Ideogram 4 was built around. That means a box doesn't *have* to be a character LoRA. It can be "a wooden table," "a neon sign in the background," "a dog sitting on the left." Krea 2 honors the placement.
2. **Identity (LoRA boxes only).** For any box you also assign a LoRA to, this node hard-masks that LoRA's effect to the box on top of the layout prompt — locking in a specific trained identity where the prompt alone can't.

So the workflow is a single, unified layout tool: sketch the whole scene as boxes, describe each one, and reinforce the boxes that need a precise identity with a LoRA. Objects and characters, all placed by the same boxes, responding the way Ideogram 4 bounding-box prompts do.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CliffNodes/ComfyUI-Krea2-Regional-MultiLoRA.git
```

Restart ComfyUI. The node appears as **Krea2 Regional Multi-LoRA** under the `Krea2` category.

**Requirements:**
- ComfyUI with Krea 2 support (a recent `master` build — needs the `Krea2` model class and `krea2_to_diffusers` LoRA key map).
- Python packages: `torch`, `safetensors` (already present in any ComfyUI install).
- No other custom-node dependencies. This node is fully standalone.

The example workflow additionally uses `Ideogram4PromptBuilderKJ` from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) as the box-drawing canvas. Any node that outputs a `BOUNDING_BOX` will work; KJNodes is just the one the example is wired for.

---

## Required Models

All from the Krea 2 release:

| Loader | File |
|--------|------|
| UNETLoader | `krea2_turbo_bf16.safetensors` (or `krea2_bf16.safetensors`) |
| CLIPLoader (type `krea2`) | `qwen3vl_4b_bf16.safetensors` |
| VAELoader | `qwen_image_vae.safetensors` |

Your **character LoRAs must be trained against Krea 2**. The reference trainer is [ai-toolkit](https://github.com/ostris/ai-toolkit). FLUX or Ideogram LoRAs will load without erroring but produce poor likeness — the attention dimensions don't line up.

---

## How to Use

The example workflow (`example_workflows/krea2_regional_multilora.json`) wires everything up. The flow is:

```
UNETLoader ─┐
CLIPLoader ─┤─► (optional global LoRA) ─► Krea2RegionalMultiLoRA ─► KSampler ─► VAEDecode ─► SaveImage
VAELoader ──┘                                     ▲
                                                  │
                     Ideogram4PromptBuilderKJ ────┘
                     (scene prompt + one box per character)
```

**Step by step:**

1. **Write your scene prompt** in the box builder. Describe the overall composition — setting, lighting, camera — *not* the individual characters. Example: *"two people standing together at an outdoor cafe, golden hour, 50mm."*

2. **Draw one box per character**, in order (left to right is the natural convention). Each box should cover roughly where that person's face and upper body will land. As you draw, **rows appear automatically** in the Krea2 node.

3. **Assign a LoRA to each row** in the Krea2 node. Row 1 → box 1, row 2 → box 2, and so on. Set each row's strength (0.8–1.2 is the usable range).

4. **Sampler settings** for Krea 2 Turbo: `euler` / `bong_tangent` / 8–12 steps / **CFG 1.0**. The negative conditioning is zeroed out — Krea 2 is designed to run at CFG 1.

5. **Queue.**

---

## Node Inputs

| Input | Notes |
|-------|-------|
| `model` / `clip` | From your loaders (or a global LoRA loader first). |
| `canvas_width` / `canvas_height` | Wire from the box builder's width/height. Used to interpret pixel-space boxes. |
| `regions_json` | The source of truth for the region list. The `+ Add Region` / `remove` buttons and box auto-sync edit this for you — you rarely touch it directly. |
| `split_mode` | `bbox` = use the drawn boxes (default). `auto_vertical` / `auto_horizontal` = split the canvas into equal strips, no boxes needed. |
| `seam_feather` | Edge softness (fraction of the token grid). `0.08` default. Raise toward `0.15` if you see hard seams. |
| `blend_override` | Keep at `0` for clean separation. Raising it mixes all LoRAs across the whole image; identities collapse past ~0.5. |
| `bboxes` (optional) | The `BOUNDING_BOX` wire from your box builder. |
| `base_strength` (optional) | Global multiplier over every region's strength. |

**Outputs:** `model` and `clip` (wire model → KSampler). The `masks` / `data` outputs are debug/preview payloads and can be left unconnected.

---

## Troubleshooting

**Characters still look merged**
Your boxes probably overlap. Shrink them so there's a clear gap. You can also lower `seam_feather` to tighten the boundaries.

**Node logs "region N matched 0 layers"**
That LoRA's key format doesn't map onto Krea 2 — it was almost certainly trained on a different model family (FLUX, SDXL, etc.). Use Krea 2 LoRAs.

**One character is right, the other is generic**
The row order doesn't match the box order. Row 1 always pairs with the first box drawn. Re-check the ordering in the box builder.

**Hard seam between regions**
Raise `seam_feather` to `0.12`–`0.15`, or overlap the boxes slightly (5–10% of canvas width) so there's a small blend zone.

**One LoRA dominates**
Equalize strengths, and check box sizes — a box twice as large gives that LoRA twice the spatial footprint even at equal strength.

---

## How It Differs From Attention-Bias Approaches

Some regional-LoRA tools steer identities using an **attention bias** — they add a penalty that *discourages* a LoRA's tokens from attending outside their region. It's a statistical nudge; the model can still route around it, which is why you get partial bleed.

This node uses **activation-delta masking** instead. The LoRA's contribution is multiplied by zero outside its box, at the point where it's added to the activations. There's nothing to route around — the contribution is gone. That's the difference between "please stay in your box" and "you cannot leave your box."

---

## License

MIT. See `LICENSE`.

## Credits

- **Node by Fedor.**
- Built on ComfyUI's model-wrapper / forward-hook system.
- Example workflow uses [`Ideogram4PromptBuilderKJ`](https://github.com/kijai/ComfyUI-KJNodes) by kijai for the box-drawing canvas.
- Krea 2 by Krea. Character LoRA training via [ai-toolkit](https://github.com/ostris/ai-toolkit) by ostris.
