"""
Qwen Image Edit Pipeline for background removal and image preprocessing.

This module provides a pipeline for using Qwen Image Edit models to preprocess
images for 3D generation, including background cleanup and view standardization.
"""
from typing import Optional
import math
import hashlib
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.models import QwenImageTransformer2DModel
from pydantic import BaseModel, Field
from loguru import logger

from generator.trellis2.pipelines.base import Pipeline
from generator.config import qwen_scheduler_settings, qwen_edit_settings


class TextPrompting(BaseModel):
    """Pydantic model for text prompting configuration."""
    prompt: str = Field(alias="positive")
    negative_prompt: Optional[str] = Field(default=None, alias="negative")
    views_prompt: Optional[str] = Field(default=None, alias="views")

    model_config = {"populate_by_name": True}


class QwenEditPipeline(Pipeline):
    """
    Pipeline for Qwen Image Edit operations.
    
    This pipeline wraps the Diffusers QwenImageEditPlusPipeline and provides
    a consistent interface for image editing operations used in preprocessing.
    
    Prompts must be supplied externally (e.g. via config/qwen_edit_prompt.json).
    """

    def __init__(
        self,
        pipe: QwenImageEditPlusPipeline,
        pipe_config: Optional[dict] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ):
        """
        Initialize the Qwen Edit Pipeline.
        
        Args:
            pipe: The underlying Diffusers pipeline.
            pipe_config: Configuration for inference (steps, cfg scale, dimensions).
            prompt: Positive prompt for editing (required — loaded from config).
            negative_prompt: Negative prompt for editing (optional — loaded from config).
        
        Raises:
            ValueError: If *prompt* is not provided.
        """
        if not prompt:
            raise ValueError(
                "A positive prompt must be provided (e.g. from config/qwen_edit_prompt.json). "
                "Hardcoded defaults have been removed to keep a single source of truth."
            )
        # Don't call super().__init__ with models since we manage our own pipeline
        self.models = {}
        self.pipe = pipe
        self.pipe_config = pipe_config or {
            "num_inference_steps": qwen_edit_settings.num_inference_steps,
            "true_cfg_scale": qwen_edit_settings.true_cfg_scale,
            "height": qwen_edit_settings.qwen_edit_height,
            "width": qwen_edit_settings.qwen_edit_width,
        }
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self._device = pipe.device if hasattr(pipe, 'device') else torch.device('cpu')

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        model_revisions: dict[str, str],
        dtype: str = "bfloat16",
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        pipe_config: Optional[dict] = None,
    ) -> "QwenEditPipeline":
        """
        Load a pretrained Qwen Edit pipeline.
        
        Args:
            path: Path to the Qwen Edit model (HuggingFace repo ID or local path).
            model_revisions: Dict mapping repo IDs to their revisions.
            dtype: Model dtype ("bfloat16", "float16", or "float32").
            prompt: Positive prompt for editing (required — loaded from config).
            negative_prompt: Negative prompt for editing (optional — loaded from config).
            pipe_config: Configuration for inference parameters.
            
        Returns:
            Initialized QwenEditPipeline instance.
            
        Raises:
            ValueError: If *prompt* is not provided.
        """
        # Get revision for the model
        repo_id = '/'.join(path.split('/')[:2])
        revision = model_revisions.get(repo_id)
        
        logger.info(f"Loading Qwen Edit Pipeline from {path} (revision: {revision})")

        # Resolve dtype
        dtype_mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        torch_dtype = dtype_mapping.get(dtype.lower(), torch.bfloat16)
        
        if not torch.cuda.is_available() and torch_dtype in {torch.float16, torch.bfloat16}:
            logger.warning("CUDA not available, falling back to float32")
            torch_dtype = torch.float32

        # Load Transformer
        logger.info("Loading Qwen transformer...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            revision=revision,
        )

        # Configure Scheduler
        scheduler_config = qwen_scheduler_settings.model_config
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Load Pipeline
        logger.info("Loading Qwen pipeline...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            path,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
            revision=revision,
        )

        # Load LoRA if specified
        if qwen_edit_settings.qwen_edit_lora_path is not None:
            logger.info(f"Using qwen edit lora: {qwen_edit_settings.qwen_edit_lora_path}")
            lora_edit_path = qwen_edit_settings.qwen_edit_lora_path
            lora_repo_id = '/'.join(lora_edit_path.split('/')[:2])
            lora_revision = model_revisions.get(lora_repo_id)
            logger.info(f"Loading LoRA from {lora_edit_path} (revision: {lora_revision})")
            pipe.load_lora_weights(
                lora_edit_path,
                weight_name=qwen_edit_settings.qwen_edit_lora_weight_filename,
                revision=lora_revision,
                adapter_name="lora_edit_adapter",
            )

        if qwen_edit_settings.qwen_edit_lora_angles_path is not None:
            logger.info(f"Using qwen edit multiple angles lora: {qwen_edit_settings.qwen_edit_lora_angles_path}")
            lora_angles_repo = qwen_edit_settings.qwen_edit_lora_angles_path
            lora_repo_id = '/'.join(lora_angles_repo.split('/')[:2])
            lora_revision = model_revisions.get(lora_repo_id)
            logger.info(f"Loading LoRA from {lora_angles_repo} (revision: {lora_revision})")
            pipe.load_lora_weights(
                lora_angles_repo,
                weight_name=qwen_edit_settings.qwen_edit_lora_angles_weight_filename,
                revision=lora_revision,
                adapter_name="lora_angle_adapter",
            )

        logger.success(f"Qwen Edit Pipeline loaded with dtype={torch_dtype}")
        
        return cls(
            pipe=pipe,
            pipe_config=pipe_config,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

    @property
    def device(self) -> torch.device:
        """Get the current device of the pipeline."""
        return self._device

    def to(self, device: torch.device) -> "QwenEditPipeline":
        """Move the pipeline to the specified device."""
        self.pipe.to(device)
        self._device = device
        return self

    def _derive_seed(self, text: str) -> int:
        """Derive a deterministic seed from text."""
        hash_object = hashlib.md5(text.encode("utf-8"))
        return int(hash_object.hexdigest()[:8], 16) % (2**32)

    def _prepare_input_image(self, image: Image.Image, megapixels: float = 1.0) -> Image.Image:
        """
        Resize image to target megapixels while maintaining aspect ratio.
        
        Args:
            image: Input PIL Image.
            megapixels: Target size in megapixels.
            
        Returns:
            Resized PIL Image (RGB mode).
        """
        # Convert to RGB if needed (Qwen model expects RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        total = int(megapixels * 1024 * 1024)
        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        true_cfg_scale: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Image.Image:
        """
        Run the Qwen edit pipeline on an input image.
        
        Args:
            image: Input PIL Image to edit.
            prompt: Text prompt override. Falls back to the instance prompt (from config).
            negative_prompt: Negative prompt override. Falls back to instance value (from config).
            seed: Random seed for reproducibility. Derived from prompt if not provided.
            num_inference_steps: Number of denoising steps.
            true_cfg_scale: Classifier-free guidance scale.
            height: Output height.
            width: Output width.
            
        Returns:
            Edited PIL Image.
        """
        # Use defaults if not provided
        prompt = prompt or self.prompt
        negative_prompt = negative_prompt or self.negative_prompt
        
        if seed is None:
            seed = self._derive_seed(prompt)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare input image
        input_image = self._prepare_input_image(image)
        
        logger.info(f"Running Qwen Edit | Seed: {seed} | Input size: {input_image.size}")

        # Build config
        config = self.pipe_config.copy()
        if num_inference_steps is not None:
            config["num_inference_steps"] = num_inference_steps
        if true_cfg_scale is not None:
            config["true_cfg_scale"] = true_cfg_scale
        if height is not None:
            config["height"] = height
        if width is not None:
            config["width"] = width

        result = self.pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            **config
        )

        output_image = result.images[0]
        logger.success(f"Qwen Edit complete | Output size: {output_image.size}")
        
        return output_image

    def __call__(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Callable interface for the pipeline."""
        return self.run(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            **kwargs,
        )
