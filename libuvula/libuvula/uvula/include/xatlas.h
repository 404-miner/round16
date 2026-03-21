/*
MIT License

Copyright (c) 2018-2020 Jonathan Young

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
/*
thekla_atlas
MIT License
https://github.com/Thekla/thekla_atlas
Copyright (c) 2013 Thekla, Inc
Copyright NVIDIA Corporation 2006 -- Ignacio Castano <icastano@nvidia.com>
*/
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <vector>

namespace xatlas
{

// A group of connected faces, belonging to a single atlas.
struct Chart
{
    uint32_t* faceArray;
    uint32_t atlasIndex; // Sub-atlas index.
    uint32_t faceCount;
    uint32_t material;
};

// Output vertex.
struct PlacedVertex
{
    int32_t atlasIndex; // Sub-atlas index. -1 if the vertex doesn't exist in any atlas.
    int32_t chartIndex; // -1 if the vertex doesn't exist in any chart.
    float uv[2]; // Not normalized - values are in Atlas width and height range.
    uint32_t xref; // Index of input vertex from which this output vertex originated.
};

// Output mesh.
struct Mesh
{
    Chart* chartArray;
    uint32_t* indexArray;
    PlacedVertex* vertexArray;
    uint32_t chartCount;
    uint32_t indexCount;
    uint32_t vertexCount;
};

// Empty on creation. Populated after charts are packed.
struct Atlas
{
    uint32_t* image;
    Mesh* meshes; // The output meshes, corresponding to each AddMesh call.
    float* utilization; // Normalized atlas texel utilization array. E.g. a value of 0.8 means 20% empty space. atlasCount in length.
    uint32_t width; // Atlas width in texels.
    uint32_t height; // Atlas height in texels.
    uint32_t atlasCount; // Number of sub-atlases. Equal to 0 unless PackOptions resolution is changed from default (0).
    uint32_t chartCount; // Total number of charts in all meshes.
    uint32_t meshCount; // Number of output meshes. Equal to the number of times AddMesh was called.
    float texelsPerUnit; // Equal to PackOptions texelsPerUnit if texelsPerUnit > 0, otherwise an estimated value to match PackOptions resolution.
};

// Create an empty atlas.
Atlas* Create();

void Destroy(Atlas* atlas);

enum class IndexFormat
{
    UInt16,
    UInt32
};

enum class AddMeshError
{
    Success, // No error.
    Error, // Unspecified error.
    IndexOutOfRange, // An index is >= MeshDecl vertexCount.
    InvalidFaceVertexCount, // Must be >= 3.
    InvalidIndexCount // Not evenly divisible by 3 - expecting triangles.
};

struct UvMeshDecl
{
    const void* vertexUvData = nullptr;
    const void* indexData = nullptr; // optional
    const uint32_t* faceMaterialData = nullptr; // Optional. Overlapping UVs should be assigned a different material. Must be indexCount / 3 in length.
    uint32_t vertexCount = 0;
    uint32_t vertexStride = 0;
    uint32_t indexCount = 0;
    int32_t indexOffset = 0; // optional. Add this offset to all indices.
    IndexFormat indexFormat = IndexFormat::UInt16;
};

AddMeshError AddUvMesh(Atlas* atlas, const UvMeshDecl& decl);

void SetCharts(Atlas* atlas, const std::vector<std::vector<size_t>>& grouped_faces);

struct PackOptions
{
    // Charts larger than this will be scaled down. 0 means no limit.
    uint32_t maxChartSize = 0;

    // Number of pixels to pad charts with.
    uint32_t padding = 0;

    // Unit to texel scale. e.g. a 1x1 quad with texelsPerUnit of 32 will take up approximately 32x32 texels in the atlas.
    // If 0, an estimated value will be calculated to approximately match the given resolution.
    // If resolution is also 0, the estimated value will approximately match a 1024x1024 atlas.
    float texelsPerUnit = 0.0f;

    // If 0, generate a single atlas with texelsPerUnit determining the final resolution.
    // If not 0, and texelsPerUnit is not 0, generate one or more atlases with that exact resolution.
    // If not 0, and texelsPerUnit is 0, texelsPerUnit is estimated to approximately match the resolution.
    uint32_t resolution = 0;

    // Leave space around charts for texels that would be sampled by bilinear filtering.
    bool bilinear = true;

    // Align charts to 4x4 blocks. Also improves packing speed, since there are fewer possible chart locations to consider.
    bool blockAlign = false;

    // Slower, but gives the best result. If false, use random chart placement.
    bool bruteForce = false;

    // Rotate charts to the axis of their convex hull.
    bool rotateChartsToAxis = true;

    // Rotate charts to improve packing.
    bool rotateCharts = true;
};

// Call after ComputeCharts. Can be called multiple times to re-pack charts with different options.
void PackCharts(Atlas* atlas, PackOptions packOptions = PackOptions());

} // namespace xatlas
