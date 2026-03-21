import argparse
from pathlib import Path
from time import time

import numpy as np
import trimesh
import pyuvula
from pyuvula import load_mesh, save_mesh


def unwrap_with_uvula(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = time()

    out_uvs, out_vertices, out_faces, _, _, _ = pyuvula.unwrap(np.array(mesh.vertices), np.array(mesh.faces))

    t2 = time()
    print(f"libuvula: {(t2 - t1)} secs")
    print(f"[AFTER UNWRAP] vertices: {out_vertices.shape}, faces: {out_faces.shape}, uvs: {out_uvs.shape}")

    return out_vertices, out_faces, out_uvs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_folder_path", type=Path)
    parser.add_argument("--output_folder", type=Path, default=Path("./unwrapped"))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mesh_folder_path = args.mesh_folder_path
    output_folder = args.output_folder

    output_folder_uvula = output_folder / "uvula"
    output_folder_uvula.mkdir(parents=True, exist_ok=True)

    for path in mesh_folder_path.iterdir():
        file_name = path.stem
        mesh = load_mesh(path)
        print("*" * 100)
        print("Processing mesh: ", path.as_posix())
        print("[BEFORE WELD] vertices: ", mesh.vertices.shape,"; faces: " , mesh.faces.shape)
        mesh.merge_vertices()
        print("[AFTER WELD] vertices: ", mesh.vertices.shape,"; faces: " , mesh.faces.shape)

        vertices, faces, uvs = unwrap_with_uvula(mesh)
        save_mesh(vertices, faces, uvs, output_folder_uvula, f"{file_name}_uvula_mesh", "glb")
        print("\n")

