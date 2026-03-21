import math
from pathlib import Path
from typing import Optional, Tuple

from pydantic_settings import BaseSettings


config_dir = Path(__file__).parent


class QwenEditSettings(BaseSettings):
    # Qwen edit settings
    qwen_edit_model_path: str = "Qwen/Qwen-Image-Edit-2511"
    qwen_edit_lora_repo: str = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    qwen_edit_lora_weight_name: str = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    lora_angles_filename: str = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
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


qwen_edit_settings = QwenEditSettings()
qwen_scheduler_settings = QwenEditSchedulerSettings()
bg_remover_settings = BGRemoverSettings()
trellis_settings = TrellisSettings()
server_settings = ServerSettings()
