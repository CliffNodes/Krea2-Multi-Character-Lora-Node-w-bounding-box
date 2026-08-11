"""End-to-end through Krea2RegionalMultiLoRAV3.apply with a stubbed model.

The stub VAE encodes an image to a constant latent equal to its mean pixel
value, so after the mold runs the denoised tensor reads back the identity of
whichever reference owns that box. That is what makes "did this character's
reference reach this character's box" directly assertable.
"""

import json

import pytest
import torch

SIGMA_IN_WINDOW = torch.tensor(0.5)
SIGMA_OUTSIDE = torch.tensor(5.0)
LATENT = (1, 16, 32, 32)


def denoise(model, sigma=SIGMA_IN_WINDOW):
    return model.post_cfg({"denoised": torch.zeros(*LATENT), "sigma": sigma})


def strip_centre(latent, index, count):
    """Mean of the middle of horizontal strip `index` of `count`, away from the
    seam - region masks feather across strip edges, so edges are not the signal."""
    h = latent.shape[2]
    lo, hi = index * h // count, (index + 1) * h // count
    pad = (hi - lo) // 4
    return latent[0, 0, lo + pad:hi - pad, :].mean().item()


def assert_owns_strip(latent, index, count, expected, others):
    """Strip `index` was molded by the reference worth `expected`.

    Asserted as "nearer its own reference than any other" plus a loose absolute
    band, rather than an exact value: neighbouring masks feather across the seam
    by a percent or two, and pinning that arithmetic would make the test a
    change-detector for the feather curve instead of for box ownership.
    """
    got = strip_centre(latent, index, count)
    assert got == pytest.approx(expected, abs=0.05)
    for other in others:
        assert abs(got - expected) < abs(got - other), (
            f"strip {index} reads {got:.3f}, closer to {other} than to its own {expected}"
        )


class TestLoraPath:
    def test_wired_loras_reach_the_sampler_with_scaled_strengths(
        self, apply_v3, build_chain, lora_dir
    ):
        chain = build_chain([
            {"lora_name": "char_A.safetensors", "strength": 1.1, "enable": True, "name": "A"},
            {"lora_name": "char_B.safetensors", "strength": 0.5, "enable": True, "name": "B"},
        ])
        (_, _, _, data), model = apply_v3(characters=chain, base_strength=2.0)

        assert [a["name"] for a in data["adapters"]] == ["A", "B"]
        assert [a["strength"] for a in data["adapters"]] == [pytest.approx(2.2), pytest.approx(1.0)]
        scales = [[e["scale"] for e in region.values()] for region in model.session.region_loras]
        assert scales == [[pytest.approx(2.2)], [pytest.approx(1.0)]]

    def test_no_wrapper_when_no_character_carries_a_lora(self, apply_v3, build_chain):
        chain = build_chain([{"lora_name": "None", "strength": 1.1, "enable": True,
                              "ref_image": torch.full((1, 8, 8, 3), 0.5)}])
        (_, _, _, _), model = apply_v3(characters=chain, vae=None)
        assert model.wrappers == []

    def test_zero_strength_character_arms_nothing(self, apply_v3, build_chain, lora_dir):
        chain = build_chain([
            {"lora_name": "char_A.safetensors", "strength": 0.0, "enable": True},
        ])
        (_, _, _, data), model = apply_v3(characters=chain)
        assert model.wrappers == []
        assert data["adapters"] == []

    def test_missing_lora_surfaces_as_a_named_error(self, apply_v3, build_chain, lora_dir):
        chain = build_chain([
            {"lora_name": "character_A.safetensors", "strength": 1.0, "enable": True},
        ])
        with pytest.raises(FileNotFoundError, match="character_A.safetensors"):
            apply_v3(characters=chain)


class TestReferencePath:
    def test_each_reference_molds_its_own_box(self, apply_v3, build_chain, vae, flat_image):
        chain = build_chain([
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.25)},
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.75)},
        ])
        _, model = apply_v3(characters=chain, vae=vae)
        out = denoise(model)
        assert_owns_strip(out, 0, 2, 0.25, others=[0.75])
        assert_owns_strip(out, 1, 2, 0.75, others=[0.25])

    def test_a_disabled_character_frees_its_box_and_the_rest_shift_up(
        self, apply_v3, build_chain, vae, flat_image
    ):
        """Boxes are claimed by ACTIVE rows in order, so disabling the middle
        character must hand its box to the one after it, not leave a hole."""
        chain = build_chain([
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.2)},
            {"lora_name": "None", "strength": 1.0, "enable": False, "ref_image": flat_image(0.5)},
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.8)},
        ])
        (_, _, _, data), model = apply_v3(characters=chain, vae=vae)
        assert len(data["adapters"]) == 2

        out = denoise(model)
        assert_owns_strip(out, 0, 2, 0.2, others=[0.5, 0.8])
        # The disabled character's 0.5 must be absent everywhere, and the third
        # character must have inherited its box rather than leaving a hole.
        assert_owns_strip(out, 1, 2, 0.8, others=[0.5, 0.2])

    def test_ref_enable_off_keeps_the_wire_but_stops_the_steering(
        self, apply_v3, build_chain, vae, flat_image
    ):
        chain = build_chain([
            {"lora_name": "None", "strength": 1.0, "enable": True,
             "ref_image": flat_image(0.25), "ref_enable": False},
        ])
        _, model = apply_v3(characters=chain, vae=vae)
        assert model.post_cfg is None

    def test_no_vae_skips_reference_guidance_without_crashing(
        self, apply_v3, build_chain, flat_image, caplog
    ):
        chain = build_chain([
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.25)},
        ])
        with caplog.at_level("WARNING"):
            _, model = apply_v3(characters=chain, vae=None)
        assert model.post_cfg is None
        assert "no VAE" in caplog.text

    def test_mold_is_inert_outside_the_sampling_window(
        self, apply_v3, build_chain, vae, flat_image
    ):
        chain = build_chain([
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.25)},
        ])
        _, model = apply_v3(characters=chain, vae=vae, ref_start_percent=0.0, ref_end_percent=0.5)
        untouched = denoise(model, SIGMA_OUTSIDE)
        assert torch.equal(untouched, torch.zeros(*LATENT))

    def test_socket_image_is_used_rather_than_a_filename(
        self, apply_v3, build_chain, vae, flat_image
    ):
        """The row's ref_image is a sentinel, not a path; nothing should try to
        read it off disk."""
        chain = build_chain([
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.6)},
        ])
        (_, _, _, data), model = apply_v3(characters=chain, vae=vae)
        assert data["adapters"][0]["ref_image"].startswith("krea2:ref:")
        assert_owns_strip(denoise(model), 0, 1, 0.6, others=[0.0])


class TestRegionsJsonFallback:
    def test_chain_overrides_whatever_was_typed_in_regions_json(
        self, apply_v3, build_chain, lora_dir
    ):
        typed = json.dumps([{"lora": "char_B.safetensors", "strength": 9.0, "enable": True}])
        chain = build_chain([
            {"lora_name": "char_A.safetensors", "strength": 1.0, "enable": True, "name": "A"},
        ])
        (_, _, _, data), _ = apply_v3(characters=chain, regions_json=typed)
        assert [a["lora"] for a in data["adapters"]] == ["char_A.safetensors"]

    def test_regions_json_still_drives_the_node_when_nothing_is_wired(
        self, apply_v3, lora_dir
    ):
        typed = json.dumps([{"lora": "char_B.safetensors", "strength": 1.0,
                             "enable": True, "name": "typed"}])
        (_, _, _, data), model = apply_v3(characters=None, regions_json=typed)
        assert [a["name"] for a in data["adapters"]] == ["typed"]
        assert model.wrappers, "LoRA from regions_json was not armed"

    def test_empty_chain_falls_back_instead_of_rendering_nothing(
        self, apply_v3, lora_dir, caplog
    ):
        typed = json.dumps([{"lora": "char_B.safetensors", "strength": 1.0,
                             "enable": True, "name": "typed"}])
        with caplog.at_level("ERROR"):
            (_, _, _, data), _ = apply_v3(characters=["junk"], regions_json=typed)
        assert [a["name"] for a in data["adapters"]] == ["typed"]
        assert "yielded no usable rows" in caplog.text


def test_passthrough_when_there_is_nothing_to_do(apply_v3, build_chain, caplog):
    chain = build_chain([{"lora_name": "None", "strength": 1.0, "enable": True}])
    with caplog.at_level("WARNING"):
        (model_out, _, masks, data), model = apply_v3(characters=chain)
    assert data["adapters"] == []
    assert masks["masks"] == {}
    assert model.wrappers == [] and model.post_cfg is None
