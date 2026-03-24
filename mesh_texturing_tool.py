import os
import yaml
import json
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory

from pathlib import Path
from PIL import Image

import trimesh
from loguru import logger
from generator.trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline
from generator.trellis2.pipelines import QwenEditPipeline
from generator.config import settings



def load_mesh(mesh_file_path: Path, strip_uvs: bool = False) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_file_path.as_posix(), force="mesh", process=False)
    if strip_uvs:
        try:
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        except Exception:
            pass
    return mesh


REQUIRED_MODELS = {
    "microsoft/TRELLIS.2-4B",
    "microsoft/TRELLIS-image-large",
    "ZhengPeng7/BiRefNet",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def load_model_versions() -> dict[str, str]:
    """Load pinned model versions from model_versions.yml."""
    versions_file = Path(__file__).parent / "model_versions.yml"
    with open(versions_file) as f:
        data = yaml.safe_load(f)["huggingface"]

    # Extract revisions and validate
    model_versions = {k: v["revision"] for k, v in data.items()}

    if missing := REQUIRED_MODELS - model_versions.keys():
        raise ValueError(f"Missing required models in model_versions.yml: {missing}")

    return model_versions


def load_prompts():
    with open(settings.qwen_edit_prompt_path, "r") as f:
        prompt_data = json.load(f)
    prompt = prompt_data.get("positive") or settings.qwen_edit_prompt
    negative_prompt = prompt_data.get("negative") or settings.qwen_edit_negative_prompt
    return prompt, negative_prompt


def startup(model_revisions):
    logger.info("Loading Qwen Edit module...")
    prompt, negative_prompt = load_prompts()
    pipe_config = {
        "num_inference_steps": settings.num_inference_steps,
        "true_cfg_scale": settings.true_cfg_scale,
        "height": settings.qwen_edit_height,
        "width": settings.qwen_edit_width,
    }
    pipeline = QwenEditPipeline.from_pretrained(
        settings.qwen_edit_model_path,
        model_revisions=model_revisions,
        lora_repo=settings.qwen_edit_lora_repo,
        lora_weight_name=settings.qwen_edit_lora_weight_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        pipe_config=pipe_config,
    )
    pipeline.to("cuda")
    logger.success("Qwen Edit module loaded")
    return pipeline


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mesh_folder", type=Path, required=True)
    arg_parser.add_argument("--image_folder", type=Path, required=True)
    arg_parser.add_argument("--output_folder", type=Path, required=True)
    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    seed = 42

    model_versions = load_model_versions()
    texture_pipe = Trellis2TexturingPipeline.from_pretrained(
        settings.trellis_model_id, model_versions, config_file="texturing_pipeline.json"
    )
    texture_pipe.to("cuda")
    qwen_edit = startup(model_versions)
    prompt, negative_prompt = load_prompts()

    sorted_mesh_paths = sorted(args.mesh_folder.iterdir(), key=lambda x: x.name.lower())
    sorted_image_paths = sorted(args.image_folder.iterdir(), key=lambda x: x.name.lower())

    for mesh_path, image_path in zip(sorted_mesh_paths, sorted_image_paths):
        logger.info(f"Processing/Texturing mesh: {mesh_path.name} with image: {image_path.name}")

        output_file_name = mesh_path.stem + "_retextured.glb"
        mesh = load_mesh(mesh_path, strip_uvs=True)
        mesh.merge_vertices()
        image = Image.open(image_path.as_posix())

        edited_image = qwen_edit.run(
            image=image,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt
        )

        output_mesh = texture_pipe.run(mesh, edited_image, texture_size=1024, uv_unwrapping_backend="uvula", rast_backend="kaolin", seed=seed)
        output_mesh.export((args.output_folder/output_file_name).as_posix(), extension_webp=True)
