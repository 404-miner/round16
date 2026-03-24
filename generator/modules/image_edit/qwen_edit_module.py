from typing import Optional
import gc
import json

import torch
from PIL import Image
from loguru import logger

from generator.config import QwenEditSettings
from generator.modules.image_edit.qwen_edit import QwenEditPipeline


class QwenEditModule:
    """Standalone Qwen image-edit module with runtime config."""

    def __init__(self, settings: QwenEditSettings, model_revisions: dict[str, str]):
        self.settings = settings
        self.model_revisions = model_revisions
        self.pipeline: Optional[QwenEditPipeline] = None
        self.prompt: Optional[str] = None
        self.negative_prompt: Optional[str] = None
        self.pipe_config: dict = {}

    def _load_prompts(self) -> None:
        with open(self.settings.qwen_edit_prompt_path, "r") as f:
            prompt_data = json.load(f)
        self.prompt = prompt_data["base"].get("positive") or self.settings.qwen_edit_prompt
        self.negative_prompt = prompt_data["base"].get("negative") or self.settings.qwen_edit_negative_prompt
        self.views_prompt = prompt_data["views"].get("positive") or self.settings.qwen_edit_views_prompt

    async def startup(self) -> None:
        logger.info("Loading Qwen Edit module...")
        self._load_prompts()
        self.pipe_config = {
            "num_inference_steps": self.settings.num_inference_steps,
            "true_cfg_scale": self.settings.true_cfg_scale,
            "height": self.settings.qwen_edit_height,
            "width": self.settings.qwen_edit_width,
        }
        self.pipeline = QwenEditPipeline.from_pretrained(
            self.settings.qwen_edit_model_path,
            model_revisions=self.model_revisions,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            pipe_config=self.pipe_config,
        )
        self.pipeline.to("cuda")
        logger.success("Qwen Edit module loaded")

    async def shutdown(self) -> None:
        """Release pipeline and trigger garbage collection."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()

    def edit_image(self, prompt_image: Image.Image, seed: int, prompt: Optional[str] = None) -> Image.Image:
        if self.pipeline is None:
            raise RuntimeError("Qwen Edit module is not loaded")
        self.ensure_on_gpu()
        return self.pipeline.run(
            image=prompt_image,
            seed=seed,
            prompt=prompt or self.prompt,
            negative_prompt=self.negative_prompt,
        )

    def offload_to_cpu(self) -> None:
        """Move the Qwen pipeline to CPU to free GPU VRAM.

        Note: The caller is responsible for running gc.collect() and
        torch.cuda.empty_cache() *after* this call so that all freed
        CUDA blocks are reclaimed in a single pass.
        """
        if self.pipeline is not None:
            self.pipeline.to("cpu")
            logger.debug("Qwen Edit offloaded to CPU")

    def ensure_on_gpu(self) -> None:
        """Move the Qwen pipeline to GPU if not already there."""
        if self.pipeline is not None and self.pipeline.device != torch.device("cuda"):
            self.pipeline.to("cuda")
            logger.debug("Qwen Edit loaded to GPU")
