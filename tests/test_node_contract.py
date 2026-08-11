"""Node registration and INPUT_TYPES invariants.

These encode two rules this package learned the hard way, both of which are
invisible at runtime and only show up as corrupted saved workflows or silently
dropped links.
"""

import pytest

REGIONAL = [
    "Krea2RegionalMultiLoRAV3",
    "Krea2RegionalMultiLoRAV9",
    "Krea2RegionalMultiLoRAV12",
]


def test_character_node_is_registered(pkg):
    assert "Krea2Character" in pkg.NODE_CLASS_MAPPINGS
    assert "Krea2Character" in pkg.NODE_DISPLAY_NAME_MAPPINGS


def test_character_node_output_contract(pkg):
    cls = pkg.NODE_CLASS_MAPPINGS["Krea2Character"]
    assert cls.RETURN_TYPES == ("KREA2_CHARACTERS",)
    assert cls.RETURN_NAMES == ("characters",)
    assert hasattr(cls, cls.FUNCTION)


def test_every_node_is_branded(pkg):
    for node in set(pkg.NODE_CLASS_MAPPINGS.values()):
        assert node.CATEGORY == "Fedor Nodes"
    for name in pkg.NODE_DISPLAY_NAME_MAPPINGS.values():
        assert name.startswith("Fedor Nodes — ")


class TestLoraDropdown:
    def test_offers_none_first_then_real_files(self, pkg, lora_dir):
        cls = pkg.NODE_CLASS_MAPPINGS["Krea2Character"]
        choices = cls.INPUT_TYPES()["required"]["lora_name"][0]
        assert choices[0] == "None"
        assert "char_A.safetensors" in choices

    def test_chain_and_image_are_optional_sockets(self, pkg):
        optional = pkg.NODE_CLASS_MAPPINGS["Krea2Character"].INPUT_TYPES()["optional"]
        assert optional["characters"][0] == "KREA2_CHARACTERS"
        assert optional["ref_image"][0] == "IMAGE"


@pytest.mark.parametrize("node_name", REGIONAL)
class TestRegionalCharactersInput:
    def test_accepts_a_characters_chain(self, pkg, node_name):
        optional = pkg.NODE_CLASS_MAPPINGS[node_name].INPUT_TYPES()["optional"]
        assert optional["characters"][0] == "KREA2_CHARACTERS"

    def test_characters_is_declared_last(self, pkg, node_name):
        """Widget order IS the workflow save format (widgets_values is a
        positional array), so a new input inserted above existing ones re-reads
        every later value in already-saved graphs."""
        optional = list(pkg.NODE_CLASS_MAPPINGS[node_name].INPUT_TYPES()["optional"])
        assert optional[-1] == "characters"

    def test_characters_is_force_input(self, pkg, node_name):
        """Without forceInput the frontend attaches a widget to the socket,
        which serialises a null into the API prompt and overrides the wired
        link - the chain then arrives as None and the LoRAs vanish."""
        optional = pkg.NODE_CLASS_MAPPINGS[node_name].INPUT_TYPES()["optional"]
        assert optional["characters"][1].get("forceInput") is True

    def test_apply_accepts_the_keyword(self, pkg, node_name):
        import inspect
        cls = pkg.NODE_CLASS_MAPPINGS[node_name]
        params = inspect.signature(getattr(cls, cls.FUNCTION)).parameters
        assert "characters" in params
        assert params["characters"].default is None, "must stay optional"


def test_v12_inherits_v9_entrypoints(pkg):
    """V12 defines neither INPUT_TYPES nor apply; wiring V9 is what covers it."""
    v9 = pkg.NODE_CLASS_MAPPINGS["Krea2RegionalMultiLoRAV9"]
    v12 = pkg.NODE_CLASS_MAPPINGS["Krea2RegionalMultiLoRAV12"]
    assert issubclass(v12, v9)
    assert "INPUT_TYPES" not in vars(v12)
    assert "apply" not in vars(v12)
