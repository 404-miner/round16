from time import time
from typing import Dict
from PIL import Image

import cv2
import numpy as np
import torch
import trimesh
import cumesh
import pyuvula
from loguru import logger
from generator.trellis2.representations.mesh.base import MeshWithVoxel
from generator.modules.post_processing.mesh_rasterizer import rasterize_mesh_attributes, rasterize_mesh_attributes_drtk
from generator.modules.post_processing.data_structures import MeshDataWithAttributeGrid, MeshData, MeshRasterizationData, AttributeGrid
from generator.modules.post_processing.mesh_texturing_utils import map_mesh_rasterization, sample_grid_attributes
from generator.config.settings import mesh_post_processing_settings, AlphaMode



class MeshPostProcessor:
    DEFAULT_AABB = torch.as_tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._settings = mesh_post_processing_settings

    def _prepare_mesh(
            self, mesh: MeshWithVoxel, aabb: torch.Tensor, compute_vertex_normals: bool = False
    ) -> MeshDataWithAttributeGrid:
        logger.debug(f"Preparing original mesh data")
        start_time = time()

        # Prepare attribute grid
        attrs = AttributeGrid(
            values=mesh.attrs.to(self._device),
            coords=mesh.coords.to(self._device),
            aabb=torch.as_tensor(aabb, dtype=torch.float32, device=self._device),
            voxel_size=torch.as_tensor(mesh.voxel_size, dtype=torch.float32, device=self._device).broadcast_to(3)
        )

        vertices = mesh.vertices.to(self._device)
        faces = mesh.faces.to(self._device)

        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(vertices, faces)
        cumesh_mesh.fill_holes(max_hole_perimeter=self._settings.max_hole_perimeter)
        logger.debug(f"After filling holes: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        vertices, faces = cumesh_mesh.read()
        vertex_normals = None

        if compute_vertex_normals:
            cumesh_mesh.compute_vertex_normals()
            vertex_normals = cumesh_mesh.read_vertex_normals()

        original_mesh_data = MeshDataWithAttributeGrid(
            vertices=vertices, faces=faces, vertex_normals=vertex_normals, attrs=attrs
        )

        # Build BVH for the current mesh to guide remeshing
        logger.debug(f"Building BVH for current mesh...")
        original_mesh_data.build_bvh()
        logger.debug(f"Done building BVH | Time: {time() - start_time:.2f}s")

        return original_mesh_data

    @staticmethod
    def _remesh(
            mesh_data: MeshDataWithAttributeGrid, decimation_target: int, mesh_band: float, project_back: float
    ) -> MeshData:
        """"""
        logger.debug("Starting remeshing")
        start_time = time()

        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(mesh_data.vertices, mesh_data.faces)

        voxel_size = mesh_data.attrs.voxel_size
        aabb = mesh_data.attrs.aabb
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

        resolution = grid_size.max().item()
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()

        # Perform Dual Contouring remeshing (rebuilds topology)
        vertices, faces = cumesh.remeshing.remesh_narrow_band_dc(
            *cumesh_mesh.read(),
            center=center,
            scale=(resolution + 3 * mesh_band) / resolution * scale,
            resolution=resolution,
            band=mesh_band,
            project_back=project_back,  # Snaps vertices back to original surface
            verbose=False,
            bvh=mesh_data.bvh,
        )
        cumesh_mesh.init(vertices, faces)
        logger.debug(f"After remeshing: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        # Simplify and clean the remeshed result
        cumesh_mesh.simplify(decimation_target, verbose=False)
        logger.debug(f"After simplifying: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces, decimation target: {decimation_target}")

        # Extract remeshed data
        vertices, faces = cumesh_mesh.read()

        logger.debug(f"Done remeshing | Time: {time() - start_time:.2f}s")
        return MeshData(
            vertices=vertices,
            faces=faces
        )

    def _cleanup(self, mesh_data: MeshDataWithAttributeGrid, decimation_target: int):
        """Cleanup and optimize the mesh using decimation and remeshing."""
        # Create cumesh from current mesh data
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(mesh_data.vertices, mesh_data.faces)

        # Step 1: Aggressive simplification (3x target)
        cumesh_mesh.simplify(decimation_target * 3, verbose=False)
        logger.debug(
            f"After initial simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        cumesh_mesh.remove_duplicate_faces()
        cumesh_mesh.repair_non_manifold_edges()
        cumesh_mesh.remove_small_connected_components(self._settings.remove_small_connected_comp_eps)
        cumesh_mesh.fill_holes(max_hole_perimeter=self._settings.max_hole_perimeter)
        logger.debug(f"After initial cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        # Step 3: Final simplification to target count
        cumesh_mesh.simplify(decimation_target, verbose=False)
        logger.debug(f"After final simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        # Step 4: Final Cleanup loop
        cumesh_mesh.remove_duplicate_faces()
        cumesh_mesh.repair_non_manifold_edges()
        cumesh_mesh.remove_small_connected_components(self._settings.remove_small_connected_comp_eps)
        cumesh_mesh.fill_holes(max_hole_perimeter=self._settings.max_hole_perimeter)
        logger.debug(f"After final cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        # Step 5: Unify face orientations
        cumesh_mesh.unify_face_orientations()

        # Extract cleaned mesh data
        vertices, faces = cumesh_mesh.read()
        return MeshData(
            vertices=vertices,
            faces=faces
        )

    def _uv_unwrap(self, mesh_data: MeshData, uv_unwrapping_backend:str) -> MeshData:
        logger.debug("Starting UV unwrapping")

        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(mesh_data.vertices, mesh_data.faces)

        if uv_unwrapping_backend == "xatlas":
            logger.debug("Using xatlas as backend.")
            xatlas_compute_charts_kwargs = {
                "max_chart_area": 1.0,
                "max_boundary_length": 2.0,
                "max_cost": 10.0,
                "normal_seam_weight": 5.0,
                "normal_deviation_weight": 1.0,
                "fix_winding": True
            }

            out_vertices, out_faces, out_uvs, out_vmaps = cumesh_mesh.uv_unwrap(
                compute_charts_kwargs={
                    "threshold_cone_half_angle_rad": self._settings.mesh_cluster_threshold_cone_half_angle,
                    "refine_iterations": self._settings.mesh_cluster_refine_iterations,
                    "global_iterations": self._settings.mesh_cluster_global_iterations,
                    "smooth_strength": self._settings.mesh_cluster_smooth_strength,
                },
                xatlas_compute_charts_kwargs=xatlas_compute_charts_kwargs,
                return_vmaps=True,
            )

            out_vertices = out_vertices.cuda()
            out_faces = out_faces.cuda()
            out_uvs = out_uvs.cuda()
            out_vmaps = out_vmaps.cuda()
            cumesh_mesh.compute_vertex_normals()
            out_normals = cumesh_mesh.read_vertex_normals()[out_vmaps]

        elif uv_unwrapping_backend == "uvula":
            logger.debug("Using uvula as backend.")

            mesh_vertices, mesh_faces = cumesh_mesh.read()
            out_uvs, out_vertices, out_faces, _, _, out_vmaps = pyuvula.unwrap(mesh_vertices.detach().cpu().numpy(),
                                                                               mesh_faces.detach().cpu().numpy())
            cumesh_mesh.compute_vertex_normals()
            out_vmaps = torch.from_numpy(out_vmaps).cuda()
            out_normals = cumesh_mesh.read_vertex_normals()[out_vmaps]
            out_vertices = torch.from_numpy(out_vertices).float().cuda()
            out_faces = torch.from_numpy(out_faces).int().cuda()
            out_uvs = torch.from_numpy(out_uvs).float().cuda()
        else:
            raise ValueError(f"Unknown unwrapping_backend was specified: {uv_unwrapping_backend}")

        return MeshData(
            vertices=out_vertices,
            faces=out_faces,
            vertex_normals=out_normals,
            uvs=out_uvs
        )

    @staticmethod
    def _rasterize(
            mesh_data: MeshData,
            original_mesh_data: MeshDataWithAttributeGrid,
            texture_size:int,
            rast_backend:str,
            use_vertex_normals: bool = False
    ):
        """Rasterize the given attributes onto the mesh UVs."""

        logger.debug("Sampling attributes(Texture rasterization)")
        if rast_backend == "kaolin":
            pos, mask = rasterize_mesh_attributes(mesh_data.uvs, mesh_data.faces, mesh_data.vertices, texture_size)
        elif rast_backend == "drtk":
            pos, mask = rasterize_mesh_attributes_drtk(mesh_data.uvs, mesh_data.faces, mesh_data.vertices, texture_size)
        else:
            raise ValueError(f"Unknown backend: {rast_backend}")

        valid_pos = pos[mask]
        valid_positions, valid_normals = valid_pos[..., :3], valid_pos[..., 3:]
        valid_normals = valid_normals if use_vertex_normals else None

        rast_data = MeshRasterizationData(
            positions=valid_positions,
            normals=valid_normals,
            mask=mask
        )

        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        rast_data = map_mesh_rasterization(rast_data, original_mesh_data)

        # Trilinear sampling from the attribute volume (Color, Material props)
        attributes = sample_grid_attributes(rast_data, original_mesh_data.attrs, texture_size)

        return attributes, rast_data

    def _post_process_texture(self, attributes: torch.Tensor, attr_layout: Dict, mask):
        logger.debug("Finalizing mesh textures")
        start_time = time()

        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attributes[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attributes[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attributes[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # Adjust alpha with gamma
        alpha = attributes[..., attr_layout['alpha']].clamp(0, 1).pow_(self._settings.alpha_gamma)
        alpha = np.clip(alpha.cpu().numpy() * 255, 0, 255).astype(np.uint8)

        mask = mask.cpu().detach().numpy()
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

        base_color_texture = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallic_roughness_texture = Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        )

        logger.debug(f"Done finalizing mesh textures | Time: {time() - start_time:.2f}s")
        return base_color_texture, metallic_roughness_texture

    def _create_textured_mesh(self, mesh_data: MeshData, base_color: Image.Image, metallic_roughness: Image.Image):
        logger.debug("Creating textured mesh")
        start_time = time()

        alpha_mode = self._settings.alpha_mode

        # Create PBR material
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_color,
            baseColorFactor=np.array([255.0, 255.0, 255.0, 255.0], dtype=np.uint8),
            metallicRoughnessTexture=metallic_roughness,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode.value,
            alphaCutoff=alpha_mode.cutoff,
            doubleSided=bool(not self._settings.remesh)
        )

        # --- Coordinate System Conversion & Final Object ---
        vertices_np = mesh_data.vertices.mul(self._settings.rescale).cpu().numpy()
        faces_np = mesh_data.faces.cpu().numpy()
        uvs_np = mesh_data.uvs.cpu().numpy()
        normals_np = mesh_data.vertex_normals.cpu().numpy()

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1]  # Flip UV V-coordinate

        textured_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        )

        logger.debug(f"Done creating textured mesh | Time: {time() - start_time:.2f}s")

        return textured_mesh

    def proces_mesh(
            self,
            mesh: MeshWithVoxel,
            texture_size: int,
            decimation_target: int,
            remesh: bool = True,
            aabb: torch.Tensor = DEFAULT_AABB,
            uv_unwrapping_backend: str = "uvula",
            rast_backend: str = "kaolin"
    ) -> trimesh.Trimesh:
        """ Post process generated mesh """
        logger.debug(f"Original mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

        mesh_data = self._prepare_mesh(mesh, aabb, True)

        if remesh:
            cleaned_mesh_data = self._remesh(
                mesh_data, decimation_target, self._settings.remesh_band, self._settings.remesh_project
            )
        else:
            cleaned_mesh_data = self._cleanup(mesh_data, decimation_target)

        cleaned_mesh_data = self._uv_unwrap(cleaned_mesh_data, uv_unwrapping_backend)
        attributes, rast_data = self._rasterize(cleaned_mesh_data, mesh_data, texture_size, rast_backend, True)
        base_color, metallic_roughness = self._post_process_texture(attributes, mesh.layout, rast_data.mask)
        textured_mesh = self._create_textured_mesh(cleaned_mesh_data, base_color, metallic_roughness)

        return textured_mesh
