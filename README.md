# 404-base-miner (Trellis 2, commercially ready to use)

<a href="https://microsoft.github.io/TRELLIS.2"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

Current base miner implementation is based on recently released TRELLIS 2 mesh generation model.
**TRELLIS.2** is a state-of-the-art large 3D generative model (4B parameters) designed for high-fidelity 
**image-to-3D** generation. It leverages a novel "field-free" sparse voxel structure termed 
**O-Voxel** to reconstruct and generate arbitrary 3D assets with complex topologies, sharp features, 
and full PBR materials.

### 🛠️ Hardware Requirements

To run this generator you will need a GPU with at least 48 GB of VRAM. It can work with GPUs from NVIDIA Blackwell family.
You can run it on Geforce 5090 RTX if the generation settings are set to 512 resolution when you call **run(...)** method 
(see **serve.py**).

### 🛠️ Software Requirements
- latest docker package (we provide docker file in "docker" folder) or latest conda environment (we provide "conda_env.yml");
- NVIDIA GPU with cuda 12.8 support
- python 3.11

### 🔑 Huggingface Token Requirement
The code needs access to the gated model (commercially compliant) on huggingface: [https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
### Installation

- Docker (building & pushing to remote register):
```console
cd /docker
docker build --build-arg GITHUB_USER="" --build-arg GITHUB_TOKEN="" -t docker_name:docker-tag .
docker tag docker_name:docker-tag docker-register-path:docker-register-name
docker push docker-register-path:docker-register-name   
```
- Conda Env. (shell script will install everything you need to run the project):
```console
# to install conda env
bash setup_env.sh

# to uninstall conda env
bash cleanup_env.sh
```

### 🚀 Usage

### How to run:
- Docker (run locally):
```commandline
docker run --gpus all -it docker_name:docker-tag bash

# outside docker
curl -X POST "http://0.0.0.0:10006/generate" -F prompt_image_file=@sample_image.png -o sample_model.glb

# optionally pass seed for reproducible generation (-1 = random)
curl -X POST "http://0.0.0.0:10006/generate" -F prompt_image_file=@sample_image.png -F seed=42 -o sample_model.glb
```
- Conda Env.:
```commandline
# start pm2 process
pm2 start generation.config.js

# view logs
pm2 logs

# send prompt image
curl -X POST "http://0.0.0.0:10006/generate" -F prompt_image_file=@sample_image.png -o sample_model.glb

# optionally pass seed for reproducible generation (-1 = random)
curl -X POST "http://0.0.0.0:10006/generate" -F prompt_image_file=@sample_image.png -F seed=42 -o sample_model.glb

# ... OR orchestrate generation 
export PROMPTS=https://github.com/404-Repo/404-active-competition/blob/main/rounds/0/prompts.txt
python generation_orchestrator.py -i $PROMPTS
```
[Competition-1 Details](https://github.com/404-Repo/404-gen-subnet/blob/main/Competitions/competition1.md)

## ⚖️ License

This model and code are released under the **[MIT License](LICENSE)**.
