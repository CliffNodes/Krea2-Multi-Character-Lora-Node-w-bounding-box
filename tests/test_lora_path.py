"""_resolve_lora_path: the guard that turned a silent typo into a named error.

A name that resolved to nothing used to be handed to safetensors as-is, which
failed several frames later as a bare FileNotFoundError and read like a bug in
the node rather than a wrong filename.
"""

import pytest


@pytest.fixture
def resolve(pkg):
    from krea2fedor.krea2_regional_multilora import _resolve_lora_path
    return _resolve_lora_path


def test_resolves_a_name_from_the_loras_folders(resolve, lora_dir):
    assert resolve("char_A.safetensors") == str(lora_dir / "char_A.safetensors")


def test_accepts_a_hand_typed_absolute_path_outside_the_loras_folders(resolve, tmp_path):
    outside = tmp_path / "elsewhere.safetensors"
    outside.write_bytes(b"not a real lora, only the path matters here")
    assert resolve(str(outside)) == str(outside)


def test_missing_lora_raises_instead_of_reaching_safetensors(resolve, lora_dir):
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve("character_A.safetensors")
    message = str(excinfo.value)
    assert "character_A.safetensors" in message, "error must name the missing file"
    assert "models/loras" in message, "error must say where it looked"


def test_error_points_at_the_dropdown(resolve, lora_dir):
    """The whole point of the Character node: names come from a real list."""
    with pytest.raises(FileNotFoundError, match="dropdown"):
        resolve("nope.safetensors")


@pytest.mark.parametrize("name", ["", "None"])
def test_empty_and_none_are_still_rejected_as_paths(resolve, lora_dir, name):
    """Rows carrying 'None' are filtered out before resolution ever happens; if
    one slips through it must fail loudly, not load a random file."""
    with pytest.raises(FileNotFoundError):
        resolve(name)
