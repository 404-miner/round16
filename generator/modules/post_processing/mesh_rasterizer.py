import torch
import kaolin
import drtk


def rasterize_mesh_attributes(uvs, faces, vertices, texture_size) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bake 3D attributes onto a 2D texture map using Kaolin.
    Replaces nvdiffrast mesh rasterization.

    Args:
        uvs: (1, V, 2) UV coordinates in range [0, 1]
        faces: (1, F, 3) Face indices
        attributes: (1, V, C) 3D attributes (e.g. positions) to interpolate
        resolution: int (H=W=resolution)

    Returns:
        interpolated: (1, H, W, C)
        mask: (1, H, W) bool
    """
    # Prepare UVs
    uvs_ndc = uvs * 2 - 1
    uvs_ndc[:, 1] = -uvs_ndc[:, 1]
    uvs_ndc = uvs_ndc.unsqueeze(0) if uvs_ndc.dim() == 2 else uvs_ndc

    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)

    faces = faces.long() if faces.dim() == 2 else faces.squeeze(0).long()

    # Index by faces
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(uvs_ndc, faces)
    face_vertex_positions = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)

    # Create 3D coordinates for rasterization
    batch_size, num_faces = face_vertices_image.shape[:2]

    # Depth values (all 0 for flat 2D rasterization)
    face_vertices_z = torch.zeros(
        (batch_size, num_faces, 3),
        device=vertices.device,
        dtype=vertices.dtype
    )

    # Rasterize with DIB-R
    with torch.no_grad():
        pos_interpolated, face_idx = kaolin.render.mesh.rasterize(
            height=texture_size,
            width=texture_size,
            face_vertices_z=face_vertices_z,
            face_vertices_image=face_vertices_image,
            face_features=face_vertex_positions,  # Interpolate vertex positions
            backend='cuda',  # Apache 2.0 licensed backend
            multiplier=1000,
            eps=1e-8
        )

    # Extract results (remove batch dimension)
    pos = pos_interpolated[0]  # (H, W, 3)
    mask = face_idx[0] >= 0  # (H, W)

    return pos, mask


def rasterize_mesh_attributes_drtk(uvs, faces, vertices, texture_size) -> tuple[torch.Tensor, torch.Tensor]:
    # Prepare UV coordinates for rasterization (rendering in UV space)
    # DRTK expects UVs in [-1, 1] range with homogeneous coordinates
    uvs_homo = torch.cat([
        uvs[:, 0:1] * texture_size - 0.5,  # x: [0,1] -> [-0.5, texture_size-0.5]
        uvs[:, 1:2] * texture_size - 0.5,  # y: [0,1] -> [-0.5, texture_size-0.5]
        torch.ones_like(uvs[:, :1])  # z-coordinate
    ], dim=-1).unsqueeze(0)  # Add batch dimension

    faces_int32 = faces.int() if faces.dtype != torch.int32 else faces

    index_img = drtk.rasterize(uvs_homo, faces_int32, texture_size, texture_size)

    # compute mesh mask. We add an additional dimention, to make it easily comatible with other BxCxHxW tensors
    mask = (index_img[0] != -1)[:, None]

    # compute differentiable depth and barycentric coordinates
    depth_img, bary_img = drtk.render(uvs_homo, faces_int32, index_img)

    # Interpolate 3D positions in UV space
    # Make sure out_vertices is 3D (drop homogeneous coordinate if present)
    vertices_3d = vertices[..., :3] if vertices.shape[-1] == 4 else vertices

    pos = drtk.interpolate(
        vertices_3d.unsqueeze(0),  # Vertex attributes [1, N, 3]
        faces_int32,  # Face indices
        index_img,  # Triangle index image
        bary_img  # Barycentric coordinates image
    )[0]

    pos = pos.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    mask = mask.squeeze(1)

    return pos, mask