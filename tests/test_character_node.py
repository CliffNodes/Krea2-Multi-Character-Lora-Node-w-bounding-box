"""Krea2Character: chain building and the JSON transport into the regional nodes."""

import json

import torch


class TestBuild:
    def test_chain_accumulates_in_wiring_order(self, build_chain):
        chain = build_chain([
            {"lora_name": "a.safetensors", "strength": 1.1, "enable": True},
            {"lora_name": "b.safetensors", "strength": 0.5, "enable": True},
        ])
        assert [c["lora"] for c in chain] == ["a.safetensors", "b.safetensors"]

    def test_autonames_by_position_and_keeps_explicit_names(self, build_chain):
        chain = build_chain([
            {"lora_name": "a.safetensors", "strength": 1.0, "enable": True},
            {"lora_name": "b.safetensors", "strength": 1.0, "enable": True, "name": "Ana"},
            {"lora_name": "c.safetensors", "strength": 1.0, "enable": True, "name": "   "},
        ])
        assert [c["name"] for c in chain] == ["character1", "Ana", "character3"]

    def test_does_not_mutate_the_upstream_chain(self, character):
        """Each node returns a new list; ComfyUI may hand the same upstream
        output to several consumers, and in-place append would corrupt them."""
        (first,) = character.build("a.safetensors", 1.0, True)
        (second,) = character.build("b.safetensors", 1.0, True, characters=first)
        assert len(first) == 1
        assert len(second) == 2

    def test_coerces_widget_types(self, character):
        (chain,) = character.build("a.safetensors", "1.25", 1, prompt="  hi  ", name=None)
        row = chain[0]
        assert row["strength"] == 1.25 and isinstance(row["strength"], float)
        assert row["enable"] is True
        assert row["prompt"] == "hi"

    def test_missing_lora_defaults_to_none(self, character):
        (chain,) = character.build(None, 1.0, True)
        assert chain[0]["lora"] == "None"

    def test_ref_image_is_carried_privately(self, character, flat_image):
        img = flat_image(0.5)
        (chain,) = character.build("a.safetensors", 1.0, True, ref_image=img)
        assert chain[0]["_ref_tensor"] is img


class TestCharactersToRegionsJson:
    def test_returns_fallback_when_nothing_wired(self, kc):
        for empty in (None, [], ()):
            assert kc.characters_to_regions_json(empty, "FALLBACK") == "FALLBACK"

    def test_emits_rows_the_shared_parser_accepts(self, kc, build_chain):
        from krea2fedor.krea2_regional_multilora_v3 import _parse_regions_v3

        chain = build_chain([
            {"lora_name": "a.safetensors", "strength": 1.1, "enable": True, "name": "A"},
            {"lora_name": "b.safetensors", "strength": 0.5, "enable": False, "name": "B"},
        ])
        parsed = _parse_regions_v3(kc.characters_to_regions_json(chain, "FB"))
        assert [(p["name"], p["lora"], p["strength"], p["enable"]) for p in parsed] == [
            ("A", "a.safetensors", 1.1, True),
            ("B", "b.safetensors", 0.5, False),
        ]

    def test_carries_v9_row_metadata_across_the_json_hop(self, kc, build_chain):
        """prompt/portrait are dropped by the shared V3 parser and re-read from
        the raw JSON by V9, so they have to survive serialisation."""
        from krea2fedor.krea2_regional_multilora_v3 import _parse_regions_v3
        from krea2fedor.krea2_regional_multilora_v9 import _apply_portrait_flags

        chain = build_chain([
            {"lora_name": "a.safetensors", "strength": 1.0, "enable": True,
             "prompt": "a man", "portrait": True},
            {"lora_name": "b.safetensors", "strength": 1.0, "enable": True,
             "prompt": "a woman"},
        ])
        raw = kc.characters_to_regions_json(chain, "FB")
        flagged = _apply_portrait_flags(_parse_regions_v3(raw), raw, False)
        assert [r["prompt"] for r in flagged] == ["a man", "a woman"]
        assert [r["portrait"] for r in flagged] == [True, False]

    def test_private_tensor_key_never_reaches_json(self, kc, character, flat_image):
        (chain,) = character.build("a.safetensors", 1.0, True, ref_image=flat_image(0.3))
        assert "_ref_tensor" not in kc.characters_to_regions_json(chain, "FB")

    def test_unwraps_a_double_wrapped_chain(self, kc, build_chain):
        """ComfyUI hands outputs over as one-element lists; a re-wrap must not
        read as 'nothing wired' and silently fall back to regions_json."""
        chain = build_chain([
            {"lora_name": "a.safetensors", "strength": 1.0, "enable": True},
            {"lora_name": "b.safetensors", "strength": 1.0, "enable": True},
        ])
        assert len(json.loads(kc.characters_to_regions_json([chain], "FB"))) == 2
        assert len(json.loads(kc.characters_to_regions_json([[chain]], "FB"))) == 2

    def test_accepts_a_bare_single_character_dict(self, kc, character):
        (chain,) = character.build("a.safetensors", 1.0, True)
        assert len(json.loads(kc.characters_to_regions_json(chain[0], "FB"))) == 1

    def test_falls_back_loudly_on_garbage(self, kc, caplog):
        """A silent fallback renders with the wrong LoRAs and looks like the
        chain 'just didn't work' - the whole failure this node exists to kill."""
        with caplog.at_level("ERROR"):
            assert kc.characters_to_regions_json(["nonsense", 42], "FB") == "FB"
        assert "yielded no usable rows" in caplog.text

    def test_skips_bad_entries_but_keeps_good_ones(self, kc, character):
        (chain,) = character.build("a.safetensors", 1.0, True, name="A")
        rows = json.loads(kc.characters_to_regions_json(chain + ["junk"], "FB"))
        assert [r["name"] for r in rows] == ["A"]


class TestRefTensorRegistry:
    def test_round_trips_the_exact_tensor(self, kc, character, flat_image):
        img = flat_image(0.42)
        (chain,) = character.build("None", 1.0, True, ref_image=img)
        row = json.loads(kc.characters_to_regions_json(chain, "FB"))[0]
        assert torch.equal(kc.lookup_ref_tensor(row["ref_image"]), img)

    def test_rows_without_an_image_get_an_empty_filename(self, kc, character):
        (chain,) = character.build("a.safetensors", 1.0, True)
        assert json.loads(kc.characters_to_regions_json(chain, "FB"))[0]["ref_image"] == ""

    def test_each_character_gets_a_distinct_key(self, kc, build_chain, flat_image):
        chain = build_chain([
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.2)},
            {"lora_name": "None", "strength": 1.0, "enable": True, "ref_image": flat_image(0.8)},
        ])
        keys = [r["ref_image"] for r in json.loads(kc.characters_to_regions_json(chain, "FB"))]
        assert len(set(keys)) == 2

    def test_real_filenames_are_not_treated_as_sentinels(self, kc):
        assert kc.lookup_ref_tensor("photo.png") is None
        assert kc.lookup_ref_tensor("") is None
        assert kc.lookup_ref_tensor(None) is None

    def test_registry_is_bounded(self, kc, flat_image):
        """Entries hold image tensors alive; the map must not grow unbounded."""
        for _ in range(kc._REF_KEEP + 50):
            kc.register_ref_tensor(flat_image(0.1, size=8))
        assert len(kc._REF_TENSORS) <= kc._REF_KEEP
