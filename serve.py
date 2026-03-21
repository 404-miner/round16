import gc
import argparse
import asyncio
from io import BytesIO
from pathlib import Path
from time import time

import yaml
import torch
import uvicorn
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI,  UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

from generator.trellis2.pipelines import secure_randint
from generator.engine import Trellis2Engine


REQUIRED_MODELS = {
    "microsoft/TRELLIS.2-4B",
    "microsoft/TRELLIS-image-large",
    "404-Gen/dinov3-vitl16-pretrain-lvd1689m",
    "ZhengPeng7/BiRefNet",
    "PramaLLC/BEN2",
    "Qwen/Qwen-Image-Edit-2511",
    "lightx2v/Qwen-Image-Edit-2511-Lightning",
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


def get_args() -> argparse.Namespace:
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


executor = ThreadPoolExecutor(max_workers=1)

class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    logger.info("Loading Trellis 2 generator models ...")
    try:
        model_versions = load_model_versions()
        logger.info(f"Loaded pinned revisions for {len(model_versions)} models")
        
        app.state.trellis2_engine = Trellis2Engine(model_versions)
        await app.state.trellis2_engine.load_models()

    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")
        raise SystemExit("Model failed to load → exiting server")

    logger.info("Warming up pipeline...")
    try:
        warmup_image_path = Path(__file__).parent / "warmup_image.png"
        warmup_image = Image.open(warmup_image_path)
        _ = generation_block(warmup_image, seed=42)
        logger.info("Warmup complete.")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM during warmup: {e}")
        clean_vram()
        raise SystemExit("GPU OOM during warmup — server cannot start reliably")
    except Exception as e:
        # Check for CuMesh OOM errors (raised as RuntimeError with "out of memory")
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda error" in error_msg:
            logger.error(f"CUDA OOM during warmup (CuMesh): {e}")
            clean_vram()
            raise SystemExit("GPU OOM during warmup — server cannot start reliably")
        logger.warning(f"Warmup failed (non-critical): {e}")
        clean_vram()

    yield

    logger.info("Shutting down...")
    if getattr(app.state, "qwen_edit", None) is not None:
        await app.state.qwen_edit.shutdown()
    clean_vram()
    logger.info("Shutdown complete")


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def generation_block(prompt_image: Image.Image, seed: int = -1):
    """ Function for 3D data generation using Qwen-edited image.
    
    Args:
        prompt_image: Input PIL image.
        seed: Random seed (-1 for random).
    
    Returns:
        BytesIO buffer with GLB data.
    """

    if seed is None or seed == -1:
        seed = secure_randint(0, 10000)
        logger.info(f"No seed provided, using random seed: {seed}")

    t_start = time()

    try:
        mesh_glb = app.state.trellis2_engine.generate_and_texture_model(
            prompt_image=prompt_image,
            seed=seed,
            trellis_texture_size=1280,
            trellis_decimation_target=800000,
            uv_unwrapping_backend="uvula",
            diff_rast_backend="kaolin",
            with_qwen_edit=False,
            with_qwen_edit_multi_view=True,
            with_adaptive_generation=False
        )

        buffer = BytesIO()
        mesh_glb.export(buffer, extension_webp=False, file_type="glb")
        buffer.seek(0)

        t_get_model = time()
        logger.debug(f"Model Generation took: {(t_get_model - t_start)} secs.")

        return buffer
    except Exception as e:
        raise RuntimeError(f"Error during model generation: {e}")


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> Response:
    """ Generates a 3D model as GLB file """

    logger.info("Task received. Prompt-Image")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image, seed)
    buffer_size = len(buffer.getvalue())
    buffer.seek(0)
    logger.info(f"Task completed.")

    async def generate_chunks():
        chunk_size = 1024 * 1024  # 1 MB
        while chunk := buffer.read(chunk_size):
            yield chunk

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={"Content-Length": str(buffer_size)}
    )


@app.get("/version", response_model=str)
async def version() -> str:
    """ Returns current endpoint version."""
    return app.version


@app.get("/health")
def health_check() -> dict[str, str]:
    """ Return if the server is alive """
    return {"status": "healthy"}


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
