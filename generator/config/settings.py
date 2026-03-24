import math
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

import numpy as np
from pydantic_settings import BaseSettings


config_dir = Path(__file__).parent


class QwenEditSettings(BaseSettings):
    # Qwen edit settings
    qwen_edit_model_path: str = "Qwen/Qwen-Image-Edit-2511"
    qwen_edit_lora_path: str = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    qwen_edit_lora_weight_filename: str = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    qwen_edit_lora_angles_path: Optional[str] = None # "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"
    qwen_edit_lora_angles_weight_filename: str = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
    qwen_edit_height: int = 1024
    qwen_edit_width: int =1024
    num_inference_steps: int = 4
    true_cfg_scale: float = 1.0
    qwen_edit_prompt_path: Path = Path(config_dir.joinpath("qwen_edit_prompt.json"))
    qwen_edit_prompt: Optional[str] = None
    qwen_edit_negative_prompt: Optional[str] = None
    qwen_edit_views_prompt: Optional[str] = None


class QwenEditSchedulerSettings(BaseSettings):
    base_image_seq_len: int = 256
    base_shift: float =  math.log(3)
    invert_sigmas: bool = False
    max_image_seq_len: int = 8192
    max_shift: float = math.log(3)
    num_train_timesteps: int = 1000
    shift: float = 1.0
    shift_terminal: Optional = None
    stochastic_sampling: bool = False
    time_shift_type:str = "exponential"
    use_beta_sigmas: bool = False
    use_dynamic_shifting: bool = True
    use_exponential_sigmas: bool = False
    use_karras_sigmas: bool = False


class TrellisSettings(BaseSettings):
    """TRELLIS.2 model configuration"""
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    sparse_structure_steps: int = 12
    sparse_structure_cfg_strength: float = 7.5
    shape_slat_steps: int = 12
    shape_slat_cfg_strength: float = 7.5 # 3.0
    tex_slat_steps: int = 12
    tex_slat_cfg_strength: float = 3.0
    pipeline_type: str = "1024_cascade"  # '512', '1024', '1024_cascade', '1536_cascade'
    max_num_tokens: int = 49152
    mode: str = "multidiffusion" # stochastic
    multiview: bool = False


class BGRemoverSettings(BaseSettings):
    input_image_size: Tuple[int, int] = (1024, 1024)
    output_image_size: Optional[Tuple[int, int]] = None
    padding_percentage: float = 0.0
    limit_padding: bool = True
    gpu: int = 0


class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 10006


class XatlasSettings(BaseSettings):
    pass


class AlphaMode(str, Enum):
    OPAQUE: str = 'OPAQUE'
    MASK: str = 'MASK'
    BLEND: str = 'BLEND'
    DITHER: str = 'DITHER'

    @property
    def cutoff(self) -> float | None:
        if self == AlphaMode.MASK or self == AlphaMode.DITHER:
            return 0.5
        elif self == AlphaMode.BLEND:
            return 0.0
        return None


class MeshPostProcessingSettings(BaseSettings):
    """GLB converter configuration"""
    decimation_target: int = 1000000
    texture_size: int = 1024
    alpha_mode: AlphaMode = AlphaMode.OPAQUE
    rescale: float = 1.0
    remesh: bool = True
    remesh_band: float = 1.0
    remesh_project: float = 0.0
    mesh_cluster_refine_iterations: int = 0
    mesh_cluster_global_iterations: int = 1
    mesh_cluster_smooth_strength: float = 1.0
    mesh_cluster_threshold_cone_half_angle: float = np.radians(90.0)
    subdivisions: int = 2
    vertex_reproject: float = 0.0
    alpha_gamma: float = 2.2
    remove_small_connected_comp_eps: float = 1e-5
    max_hole_perimeter: float = 3e-2


qwen_edit_settings = QwenEditSettings()
qwen_scheduler_settings = QwenEditSchedulerSettings()
bg_remover_settings = BGRemoverSettings()
trellis_settings = TrellisSettings()
mesh_post_processing_settings = MeshPostProcessingSettings()
server_settings = ServerSettings()
