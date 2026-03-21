// (c) 2025, UltiMaker -- see LICENCE for details

#pragma once

#include <cstdint>
#include <vector>

#include "Face.h"

class Point3F;
struct Point2F;

/*!
 * Groups, projects and packs the faces of the input mesh to non-overlapping and properly distributed UV coordinates patches.
 * Vertices shared between different chart groups are duplicated so that each chart has exclusive vertex ownership,
 * ensuring proper UV separation. The output vertices, faces and uv_coords arrays are all consistently indexed.
 * @param vertices Input/output list of vertex positions; duplicated boundary vertices are appended
 * @param faces Input/output list of faces; indices are updated to reference duplicated vertices where needed
 * @param uv_coords Output list of UV coordinates, sized to match the (potentially expanded) vertices array
 * @param texture_width Output width to be used for the texture image
 * @param texture_height Output height to be used for the texture image
 * @return
 */
bool smartUnwrap(std::vector<Point3F>& vertices, std::vector<Face>& faces, std::vector<Point2F>& uv_coords,
                 uint32_t& texture_width, uint32_t& texture_height, std::vector<uint32_t>& vertex_remap);