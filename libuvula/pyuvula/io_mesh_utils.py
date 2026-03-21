from pathlib import Path

import trimesh
import numpy as np


def assemble_textured_mesh(V: np.ndarray, F: np.ndarray, UV: np.ndarray) -> trimesh.Trimesh:
    # Keep indices as-is; don't let trimesh merge/simplify
    vis = trimesh.visual.texture.TextureVisuals(uv=UV[:, :2])
    return trimesh.Trimesh(vertices=V, faces=F, visual=vis, process=False)


def save_mesh(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, output_path: Path, file_name: str, file_ext:str) -> None:
    mesh = assemble_textured_mesh(vertices, faces, uv)
    output_file = output_path / (file_name + "." + file_ext)
    mesh.export(output_file.as_posix(), file_type=file_ext, include_normals=False)


def load_mesh(mesh_file_path: Path, strip_uvs: bool = False) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_file_path.as_posix(), force="mesh", process=False)
    if strip_uvs:
        try:
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
        except Exception:
            pass

    return mesh
