import gc
import random
from PIL import Image
from typing import Optional
from time import time

import torch
import numpy as np
from trimesh import Trimesh
from loguru import logger
from generator.trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from generator.trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline
from generator.modules.image_edit.qwen_edit_module import QwenEditModule
from generator.modules.post_processing.mesh_postprocessing import MeshPostProcessor
from generator.modules.background_removal import BEN2BackgroundRemovalService, BirefNetBackgroundRemovalService
from generator.config import qwen_edit_settings, trellis_settings


class Trellis2Engine:
    def __init__(self, model_versions: dict) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_versions = model_versions
        self._trellis2_model: Trellis2ImageTo3DPipeline | None = None
        self._trellis2_texturing_model: Trellis2TexturingPipeline | None = None
        self._mesh_post_processor = MeshPostProcessor()
        self._qwen_edit_model = QwenEditModule(qwen_edit_settings, self._model_versions)
        self._bg_remover: Optional[BEN2BackgroundRemovalService | BirefNetBackgroundRemovalService | None] = None

    async def load_models(
            self, bg_removal_pipe: str = "birefnet", texturing_pipe_only: bool = False, enable_qwen_pipe: bool = True
    ) -> None:
        """"""

        if enable_qwen_pipe:
            await self._qwen_edit_model.startup()

        if bg_removal_pipe == "ben2":
            self._bg_remover = BEN2BackgroundRemovalService(self._model_versions)
        elif bg_removal_pipe == "birefnet":
            self._bg_remover = BirefNetBackgroundRemovalService(self._model_versions)
        else:
            raise ValueError(f"Unsupported bg_removal_pipe: {bg_removal_pipe}")

        await self._bg_remover.startup()

        if texturing_pipe_only:
            self._trellis2_texturing_model = Trellis2TexturingPipeline.from_pretrained(
                trellis_settings.model_id, self._model_versions
            )
            self._trellis2_texturing_model.to(self._device)
        else:
            self._trellis2_model = Trellis2ImageTo3DPipeline.from_pretrained(
                trellis_settings.model_id, self._model_versions
            )
            self._trellis2_model.to(self._device)

    async def unload_models(self) -> None:
        """"""
        del self._trellis2_model
        del self._trellis2_texturing_model

        await self._qwen_edit_model.shutdown()
        del self._qwen_edit_model

        await self._bg_remover.shutdown()
        del self._bg_remover

        gc.collect()
        torch.cuda.empty_cache()

    def _set_random_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _generate_multiple_views(self, prompt_image: Image.Image, seed: int) -> list[Image.Image]:
        """"""
        views_prompts = self._qwen_edit_model.views_prompt
        edited_images = []
        for prompt in views_prompts:
            logger.debug(f"Editing view with prompt: {prompt}")
            image = self._qwen_edit_model.edit_image(
                prompt_image=prompt_image,
                seed=seed,
                prompt=prompt
            )
            edited_images.append(image)
        edited_images.append(prompt_image.copy())  # Original image
        return edited_images

    def _get_dynamic_glb_params(self, face_count: int,  elapsed_time: float, target_time: float = 78.0) -> tuple[int, int]:
        """
        Intelligent GLB parameter selection based on remaining time budget.
        Fast tasks get higher quality processing (texture, decimation).
        Slow tasks get reduced params to stay within timeout.
        """
        TIME_TARGET = target_time
        remaining = TIME_TARGET - elapsed_time

        if remaining > 55:
            texture_size=3072,
            decimation_target=400000
            logger.debug(
                f"Dynamic GLB: FAST ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=3072, decim=400k"
            )
        elif remaining > 40:
            texture_size=2560,
            decimation_target=300000
            logger.debug(
                f"Dynamic GLB: NORMAL ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=2560, decim=300k"
            )
        elif remaining > 25:
            texture_size=1536
            decimation_target=250000
            logger.debug(
                f"Dynamic GLB: MODERATE ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> defaults"
            )
        else:
            texture_size=1536,
            decimation_target=180000
            logger.debug(
                f"Dynamic GLB: SLOW ({elapsed_time:.0f}s used, {remaining:.0f}s left, {face_count} faces) -> tex=1536, decim=180k"
            )

        return texture_size, decimation_target

    def generate_and_texture_model(
            self,
            prompt_image: Image.Image,
            seed: int,
            trellis_pipeline_type: Optional[str] = None,
            trellis_texture_size: int = 1024,
            trellis_decimation_target: int = 1000000,
            uv_unwrapping_backend: str = "uvula",
            diff_rast_backend: str = "drtk",
            with_qwen_edit: bool = False,
            with_qwen_edit_multi_view: bool = True,
            with_adaptive_generation: bool = False
    ) -> Trimesh:
        """"""
        if seed >= 0:
            self._set_random_seed(seed)

        t1 = time()
        if with_qwen_edit:
            edited_image = self._qwen_edit_model.edit_image(prompt_image, seed)
            edited_image_no_bg = self._bg_remover.remove_background([edited_image])[0]
        elif with_qwen_edit_multi_view:
            edited_images = self._generate_multiple_views(prompt_image, seed)
            edited_images_no_bg = self._bg_remover.remove_background(edited_images)
        else:
            edited_image = prompt_image
            edited_image_no_bg = self._bg_remover.remove_background([edited_image])[0]
        t2 = time()
        image_edit_time = t2 - t1
        logger.info(f"Image Editing stage took: {image_edit_time} seconds")

        self._qwen_edit_model.offload_to_cpu()

        if with_qwen_edit_multi_view:
            raw_mesh = self._trellis2_model.run_multi_image(
                images=edited_images_no_bg,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": trellis_settings.sparse_structure_steps,
                    "guidance_strength": trellis_settings.sparse_structure_cfg_strength,
                },
                shape_slat_sampler_params={
                    "steps": trellis_settings.shape_slat_steps,
                    "guidance_strength": trellis_settings.shape_slat_cfg_strength,
                },
                tex_slat_sampler_params={
                    "steps": trellis_settings.tex_slat_steps,
                    "guidance_strength": trellis_settings.tex_slat_cfg_strength,
                },
                pipeline_type=trellis_settings.pipeline_type if trellis_pipeline_type is None else trellis_pipeline_type,
                max_num_tokens=trellis_settings.max_num_tokens
            )[0]
        else:
            raw_mesh = self._trellis2_model.run(
                image=edited_image_no_bg,
                seed=seed,
                preprocess_image=False,
                pipeline_type=trellis_settings.pipeline_type if trellis_pipeline_type is None else trellis_pipeline_type,
                sparse_structure_sampler_params={
                    "steps": trellis_settings.sparse_structure_steps,
                    "guidance_strength": trellis_settings.sparse_structure_cfg_strength,
                },
                shape_slat_sampler_params={
                    "steps": trellis_settings.shape_slat_steps,
                    "guidance_strength": trellis_settings.shape_slat_cfg_strength,
                },
                tex_slat_sampler_params={
                    "steps": trellis_settings.tex_slat_steps,
                    "guidance_strength": trellis_settings.tex_slat_cfg_strength,
                },
            )[0]
        raw_mesh.simplify()
        t3 = time()
        mesh_generation_time = t3 - t2
        logger.info(f"Mesh generation took: {mesh_generation_time} seconds")

        if with_adaptive_generation:
            elapsed_time = image_edit_time + mesh_generation_time
            trellis_texture_size, trellis_decimation_target = self._get_dynamic_glb_params(raw_mesh.read_faces.shape[0], elapsed_time)

        mesh_glb = self._mesh_post_processor.proces_mesh(
            mesh=raw_mesh,
            texture_size=trellis_texture_size,
            decimation_target=trellis_decimation_target,
            remesh=True,
            rast_backend=diff_rast_backend,
            uv_unwrapping_backend=uv_unwrapping_backend
        )

        self._qwen_edit_model.ensure_on_gpu()

        return mesh_glb

    def texture_model(
            self,
            mesh: Trimesh,
            prompt_image: Image.Image,
            seed: int,
            trellis_texture_size: Optional[int] = None,
            uv_unwrapping_backend: str = "uvula",
            diff_rast_backend: str = "drtk",
            with_qwen_edit: bool = True,
    ) -> Trimesh:
        """"""
        if with_qwen_edit:
            edited_image = self._qwen_edit_model.edit_image(prompt_image, seed)[0]
            edited_image = self._bg_remover.remove_background([edited_image])
        else:
            edited_image = prompt_image
            edited_image = self._bg_remover.remove_background(edited_image)

        mesh_glb = self._trellis2_texturing_model.run(
            mesh,
            edited_image,
            texture_size=trellis_texture_size,
            uv_unwrapping_backend=uv_unwrapping_backend,
            rast_backend=diff_rast_backend,
            seed=seed
        )

        return mesh_glb
