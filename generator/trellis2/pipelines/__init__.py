import importlib
import os


def secure_randint(low: int, high: int) -> int:
    """Return a random integer in [low, high] using os.urandom."""
    range_size = high - low + 1
    num_bytes = 4
    max_int = 2 ** (8 * num_bytes) - 1

    while True:
        rand_bytes = os.urandom(num_bytes)
        rand_int = int.from_bytes(rand_bytes, "big")
        if rand_int <= max_int - (max_int % range_size):
            return low + (rand_int % range_size)


__attributes = {
    "Trellis2ImageTo3DPipeline": "trellis2_image_to_3d",
    "Trellis2TexturingPipeline": "trellis2_texturing",
    "QwenEditPipeline": "qwen_edit",
}

__submodules = ['samplers']

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, model_revisions: dict[str, str]):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
        model_revisions: Dict mapping repo IDs to their revisions for reproducibility.
    """
    import os
    import json
    
    # Get revision for the main repo
    main_repo_id = '/'.join(path.split('/')[:2])
    revision = model_revisions.get(main_repo_id)
    
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json", revision=revision)

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path, model_revisions=model_revisions)


# For PyLance
if __name__ == '__main__':
    from . import samplers
    from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from .trellis2_texturing import Trellis2TexturingPipeline
