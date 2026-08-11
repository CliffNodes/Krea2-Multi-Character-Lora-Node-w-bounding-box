"""Shared fixtures.

The package directory name contains a dash, so it cannot be imported normally;
it is loaded from its __init__.py under the stable alias ``krea2fedor``. Going
through __init__.py on purpose: that exercises the real registration path
(NODE_CLASS_MAPPINGS, the CATEGORY rewrite) rather than importing modules
piecemeal.

Helpers are exposed as fixtures rather than module-level functions because
pytest runs here in importlib import mode (see pytest.ini), where test modules
cannot ``from conftest import ...``.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent          # custom_nodes/<pkg> -> ComfyUI root

LATENT_CHANNELS = 16


def _load_package():
    if "krea2fedor" in sys.modules:
        return sys.modules["krea2fedor"]
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    spec = importlib.util.spec_from_file_location(
        "krea2fedor", REPO / "__init__.py", submodule_search_locations=[str(REPO)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["krea2fedor"] = module      # must precede exec for relative imports
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def pkg():
    return _load_package()


@pytest.fixture(scope="session")
def kc(pkg):
    """The krea2_character module."""
    import krea2fedor.krea2_character as module
    return module


@pytest.fixture(scope="session")
def lora_dir(tmp_path_factory):
    """A private loras folder, so tests never depend on the user's models."""
    _load_package()
    import folder_paths
    import safetensors.torch

    d = tmp_path_factory.mktemp("loras")
    for name, fill in (("char_A", 0.1), ("char_B", 0.2)):
        safetensors.torch.save_file({
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": torch.full((4, 32), fill),
            "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": torch.full((32, 4), fill),
            "lora_unet_double_blocks_0_img_attn_qkv.alpha": torch.tensor(4.0),
        }, str(d / f"{name}.safetensors"))
    folder_paths.add_model_folder_path("loras", str(d))
    return d


class StubModelSampling:
    def percent_to_sigma(self, percent):
        return 1.0 - float(percent)


class StubInner:
    def process_latent_in(self, latent):
        return latent


class StubModel:
    """Just enough ModelPatcher surface for the regional and reference paths."""

    def __init__(self):
        self.model = StubInner()
        self.wrappers = []
        self.post_cfg = None

    def clone(self):
        return self

    def get_model_object(self, _name):
        return StubModelSampling()

    def add_wrapper_with_key(self, _enum, key, fn):
        self.wrappers.append((key, fn))

    def set_model_sampler_post_cfg_function(self, fn):
        self.post_cfg = fn

    @property
    def session(self):
        """The _RegionalSession captured by the registered wrapper closure."""
        assert self.wrappers, "no regional wrapper was armed"
        return self.wrappers[0][1].__closure__[0].cell_contents


class StubVAE:
    """Encodes an image to a constant latent equal to its mean pixel value, so a
    mold's identity is readable straight off the denoised tensor."""

    def encode(self, pixels):
        return torch.full((1, LATENT_CHANNELS, 8, 8), float(pixels.mean()))


@pytest.fixture
def model():
    return StubModel()


@pytest.fixture
def vae():
    return StubVAE()


@pytest.fixture
def character(pkg):
    return pkg.NODE_CLASS_MAPPINGS["Krea2Character"]()


@pytest.fixture
def v3(pkg):
    return pkg.NODE_CLASS_MAPPINGS["Krea2RegionalMultiLoRAV3"]()


@pytest.fixture
def flat_image():
    """A uniform IMAGE tensor whose mean is `value` -> a mold of known identity."""
    def _make(value, size=64):
        return torch.full((1, size, size, 3), float(value))
    return _make


@pytest.fixture
def build_chain(character):
    """Chain N characters the way wiring N Krea2Character nodes would."""
    def _build(specs):
        chain = None
        for spec in specs:
            (chain,) = character.build(characters=chain, **spec)
        return chain
    return _build


@pytest.fixture
def apply_v3(v3, model):
    """v3.apply with sane defaults; overrides win. Returns (outputs, model)."""
    def _apply(**overrides):
        kwargs = dict(
            model=model, clip=None, canvas_width=512, canvas_height=512,
            regions_json="[]", split_mode="auto_horizontal", seam_feather=0.0,
            blend_override=0.0, ref_strength=1.0, ref_start_percent=0.0,
            ref_end_percent=1.0, ref_feather=0.0,
        )
        kwargs.update(overrides)
        return v3.apply(**kwargs), model
    return _apply
