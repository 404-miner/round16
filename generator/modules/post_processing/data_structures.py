from typing import Optional

import torch
import cumesh
from pydantic import BaseModel, ConfigDict


class AttributeGrid(BaseModel):
    values: torch.Tensor  # (N, K) attribute values for N voxels
    coords: torch.Tensor  # (N, 3) voxel coordinates on the grid
    aabb: torch.Tensor  # (2, 3) axis-aligned bounding box (optional)
    voxel_size: torch.Tensor  # (3,) size of voxel in each dimension

    @property
    def grid_size(self) -> torch.Tensor:
        return ((self.aabb[1] - self.aabb[0]) / self.voxel_size).round().int()

    def dense_shape(self, with_batch_size: bool = True) -> torch.Size:
        batch_size = (1,) if with_batch_size else ()
        return torch.Size(batch_size + (self.values.shape[1], *self.grid_size.tolist()))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MeshData(BaseModel):
    """Mesh geometry with vertices, faces, vertex normals and UVs."""
    vertices: torch.Tensor                            # (V, 3) vertex positions
    faces: torch.Tensor                             # (F, 3) face indices
    vertex_normals: Optional[torch.Tensor] = None     # (V, 3) vertex normals (optional)
    uvs: Optional[torch.Tensor] = None                # (V, 2) UV coordinates (optional)
    bvh: Optional[cumesh.cuBVH] = None     # BVH tree for ray tracing and projection

    def build_bvh(self):
        self.bvh = cumesh.cuBVH(self.vertices, self.faces)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MeshDataWithAttributeGrid(MeshData):
    attrs: Optional[AttributeGrid] = None


class MeshRasterizationData(BaseModel):
    positions: torch.Tensor                 # (N_valid, 3) position of mesh for each valid pixel
    normals: Optional[torch.Tensor] = None  # (N_valid, 3) surface normal for each valid pixel
    mask: Optional[torch.Tensor]=None

    # @property
    # def mask(self) -> torch.Tensor:
    #     return self.face_ids.ge(0)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AttributesMasked(BaseModel):
    values: torch.Tensor    # (N, K) attribute values for N valid pixels
    mask: torch.Tensor      # annotes which pixels are valid via boolean value

    def dense_shape(self, with_batch_size: bool = True) -> torch.Size:
        batch_size = (1,) if with_batch_size else ()
        return torch.Size(batch_size + (*self.mask.shape, self.values.shape[-1]))

    def to_dense(self, invalid: float = 0.0) -> torch.Tensor:
        size = self.dense_shape(with_batch_size=False)
        dense = self.values.new_full(size, fill_value=invalid)
        dense[self.mask] = self.values
        return dense

    model_config = ConfigDict(arbitrary_types_allowed=True)