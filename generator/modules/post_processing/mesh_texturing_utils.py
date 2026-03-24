import torch
from generator.modules.post_processing.data_structures import (MeshData,
                                                               AttributeGrid,
                                                               MeshRasterizationData,
                                                               AttributesMasked)
from flex_gemm.ops.grid_sample import grid_sample_3d


def map_mesh_rasterization(
        rast_data: MeshRasterizationData,
        mesh_data: MeshData,
        flip_vertex_normals: bool = False
) -> MeshRasterizationData:

    bvh = mesh_data.bvh
    assert bvh is not None, "Mesh BVH needs to be build for mapping"
    valid_pos = rast_data.positions

    # Map these positions back to the *original* high-res mesh to get accurate attributes
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    tris = mesh_data.faces[face_id.long()]
    tri_verts = mesh_data.vertices[tris]  # (N_new, 3, 3)
    valid_positions = (tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    valid_normals = None

    if mesh_data.vertex_normals is not None:
        tri_norms = mesh_data.vertex_normals[tris]
        valid_normals = (tri_norms * uvw.unsqueeze(-1)).sum(dim=1)

        if flip_vertex_normals:
            flip_sign = (rast_data.normals * valid_normals).sum(dim=-1, keepdim=True).sign()
            valid_normals.mul_(flip_sign)

    return MeshRasterizationData(positions=valid_positions, normals=valid_normals, mask=rast_data.mask)


def sample_grid_attributes(rast_data: MeshRasterizationData, grid: AttributeGrid, texture_size: int) -> torch.Tensor:
    voxel_size = grid.voxel_size
    aabb = grid.aabb
    coords = grid.coords
    attr_volume = grid.values

    valid_pos = rast_data.positions
    mask = rast_data.mask.to(grid.values.device)

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=grid.dense_shape(),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear'
    )
    # ).squeeze(0)

    return attrs