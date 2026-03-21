// (c) 2025, UltiMaker -- see LICENCE for details

#include "unwrap.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <queue>
#include <set>

#include <range/v3/algorithm/partition.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/map.hpp>
#include <spdlog/spdlog.h>

#include "Matrix33F.h"
#include "Point2F.h"
#include "Point3F.h"
#include "Vector3F.h"
#include "geometry_utils.h"
#include "xatlas.h"


struct FaceData
{
    const Face* face;
    size_t face_index;
    Vector3F normal;
};

/*!
 * Calculate the best projection normals according to the given input faces
 * @param faces_data The faces data
 * @return A list of normals that are far enough from each other
 */
std::vector<Vector3F> calculateProjectionNormals(const std::vector<FaceData>& faces_data)
{
    constexpr float group_angle_limit = 20.0;

    const float group_angle_limit_cos = std::cos(geometry_utils::deg2rad(group_angle_limit));
    const float group_angle_limit_half_cos = std::cos(geometry_utils::deg2rad(group_angle_limit / 2));

    // First group will be based on the normal of the very first face
    const Vector3F* project_normal = &faces_data.front().normal;

    std::vector<Vector3F> projection_normals;

    // Create an internal list containing pointers to all the faces data, it will be reorganized
    std::vector<const FaceData*> faces_to_process(faces_data.size());
    std::transform(
        faces_data.begin(),
        faces_data.end(),
        faces_to_process.begin(),
        [](const FaceData& face_data)
        {
            return &face_data;
        });

    using FaceDataIterator = std::vector<const FaceData*>::iterator;
    struct FaceDataRange
    {
        FaceDataIterator begin;
        FaceDataIterator end;
    };

    // The unprocessed_faces is a sub-range of the faces list, that contains all the faces that have not been assigned to a group yet.
    FaceDataRange unprocessed_faces = { .begin = std::next(faces_to_process.begin()), .end = faces_to_process.end() };

    while (true)
    {
        // Get all the faces that belong to the group of the current projection normal,
        // by placing them at the beginning of the unprocessed faces
        FaceDataRange current_faces_group{ .begin = unprocessed_faces.begin };
        current_faces_group.end = ranges::partition(
            unprocessed_faces.begin,
            unprocessed_faces.end,
            [&project_normal, &group_angle_limit_half_cos](const FaceData* face_data)
            {
                return face_data->normal.dot(*project_normal) > group_angle_limit_half_cos;
            });

        // All the faces placed to the current group are now no more in the unprocessed faces
        unprocessed_faces.begin = current_faces_group.end;

        // Sum all the normals of the current faces group to get the average direction
        Vector3F summed_normals = std::accumulate(
            current_faces_group.begin,
            current_faces_group.end,
            *project_normal,
            [](const Vector3F& normal, const FaceData* face_data)
            {
                return normal + face_data->normal;
            });
        if (summed_normals.normalize()) [[likely]]
        {
            projection_normals.push_back(summed_normals);
        }

        // For the next iteration, try to find the most different remaining normal from all generated normals
        float best_outlier_angle = std::numeric_limits<float>::max();
        FaceDataIterator best_outlier_face = faces_to_process.end();

        for (auto iterator = unprocessed_faces.begin; iterator != unprocessed_faces.end; ++iterator)
        {
            float face_best_angle = std::numeric_limits<float>::lowest();
            for (const Vector3F& projection_normal : projection_normals)
            {
                face_best_angle = std::max(face_best_angle, projection_normal.dot((*iterator)->normal));
            }

            if (face_best_angle < best_outlier_angle)
            {
                best_outlier_angle = face_best_angle;
                best_outlier_face = iterator;
            }
        }

        if (best_outlier_angle < group_angle_limit_cos)
        {
            // Take the normal of the best outlier as the base for the iteration of the next group
            project_normal = &(*best_outlier_face)->normal;

            // Remove the faces from the unprocessed faces
            std::iter_swap(best_outlier_face, unprocessed_faces.begin);
            ++unprocessed_faces.begin;
        }
        else if (! projection_normals.empty())
        {
            break;
        }
    }

    return projection_normals;
}

static std::vector<FaceData> makeFacesData(const std::vector<Point3F>& vertices, const std::vector<Face>& faces)
{
    std::vector<FaceData> faces_data;
    faces_data.reserve(faces.size());

    for (const auto& [index, face] : faces | ranges::views::enumerate)
    {
        const Point3F& v1 = vertices[face.i1];
        const Point3F& v2 = vertices[face.i2];
        const Point3F& v3 = vertices[face.i3];

        const std::optional<Vector3F> triangle_normal = geometry_utils::triangleNormal(v1, v2, v3);
        if (triangle_normal.has_value())
        {
            faces_data.push_back(FaceData{ &face, index, triangle_normal.value() });
        }
    }

    return faces_data;
}

/*!
 * Groups the faces that have a similar normal, and project their points as raw UV coordinates along this normal
 * @param vertices The list of vertices positions
 * @param faces The list of faces we want to project
 * @param uv_coords The UV coordinates, which should be properly sized but the input content doesn't matter. As output, they will be filled with
 *                  raw UV coordinates that overlap and are not in the [0,1] range
 * @return A list containing grouped indices of faces
 */
static std::vector<std::vector<size_t>> makeCharts(const std::vector<Point3F>& vertices, const std::vector<Face>& faces, std::vector<Point2F>& uv_coords)
{
    const std::vector<FaceData> faces_data = makeFacesData(vertices, faces);
    if (faces_data.empty()) [[unlikely]]
    {
        return {};
    }

    // Calculate the best normals to group the faces
    const std::vector<Vector3F> project_normal_array = calculateProjectionNormals(faces_data);
    if (project_normal_array.empty()) [[unlikely]]
    {
        return {};
    }

    // For each face, find the best projection normal and make groups
    std::map<const Vector3F*, std::vector<const FaceData*>> projected_faces_groups;
    for (const FaceData& face_data : faces_data)
    {
        const Vector3F* best_projection_normal = nullptr;
        float angle_best = std::numeric_limits<float>::lowest();

        for (const Vector3F& projection_normal : project_normal_array)
        {
            const float angle = face_data.normal.dot(projection_normal);
            if (angle > angle_best)
            {
                angle_best = angle;
                best_projection_normal = &projection_normal;
            }
        }

        projected_faces_groups[best_projection_normal].push_back(&face_data);
    }

    // Now project each faces according to the closest matching normal and create indices groups
    std::vector<std::vector<size_t>> grouped_faces_indices;
    for (const auto& [normal, faces_data] : projected_faces_groups)
    {
        const Matrix33F axis_mat = Matrix33F::makeOrthogonalBasis(*normal);
        std::vector<size_t> faces_group;
        faces_group.reserve(faces_data.size());

        for (const FaceData* face_from_group : faces_data)
        {
            faces_group.push_back(face_from_group->face_index);

            for (const uint32_t vertex_index : { face_from_group->face->i1, face_from_group->face->i2, face_from_group->face->i3 })
            {
                uv_coords[vertex_index] = axis_mat.project(vertices[vertex_index]);
            }
        }

        grouped_faces_indices.push_back(std::move(faces_group));
    }

    return grouped_faces_indices;
}

/*!
 * Splits each normal-group into connected components using edge adjacency (BFS).
 * Two faces are considered connected only if they share a full edge (two vertices),
 * not merely a single vertex.  This avoids the problem where the old vertex-based
 * connectivity merged distant faces that touched only at a corner.
 * @param grouped_faces Contains the grouped indices of faces from makeCharts
 * @param faces Position-merged faces for adjacency detection (@sa groupSimilarVertices)
 * @return Grouped faces with groups containing only edge-connected faces
 */
std::vector<std::vector<size_t>> splitNonLinkedFacesCharts(const std::vector<std::vector<size_t>>& grouped_faces, const std::vector<Face>& faces)
{
    std::vector<std::vector<size_t>> result;

    for (const std::vector<size_t>& faces_group : grouped_faces)
    {
        if (faces_group.size() <= 1)
        {
            result.push_back(faces_group);
            continue;
        }

        // Build local index: global face index -> local index within this group
        std::map<size_t, size_t> global_to_local;
        for (size_t i = 0; i < faces_group.size(); i++)
            global_to_local[faces_group[i]] = i;

        // Build edge -> local face list map
        using Edge = std::pair<uint32_t, uint32_t>;
        std::map<Edge, std::vector<size_t>> edge_locals;
        for (size_t li = 0; li < faces_group.size(); li++)
        {
            const Face& f = faces[faces_group[li]];
            const uint32_t v[3] = { f.i1, f.i2, f.i3 };
            for (int e = 0; e < 3; e++)
            {
                uint32_t a = v[e], b = v[(e + 1) % 3];
                if (a > b)
                    std::swap(a, b);
                edge_locals[{ a, b }].push_back(li);
            }
        }

        // Build edge-based adjacency list (local indices)
        std::vector<std::vector<size_t>> adj(faces_group.size());
        for (const auto& [edge, locals] : edge_locals)
            for (size_t i = 0; i < locals.size(); i++)
                for (size_t j = i + 1; j < locals.size(); j++)
                {
                    adj[locals[i]].push_back(locals[j]);
                    adj[locals[j]].push_back(locals[i]);
                }

        // BFS to find connected components
        std::vector<int> component(faces_group.size(), -1);
        int n_components = 0;

        for (size_t seed = 0; seed < faces_group.size(); seed++)
        {
            if (component[seed] != -1)
                continue;

            const int comp = n_components++;
            std::queue<size_t> queue;
            queue.push(seed);
            component[seed] = comp;

            while (! queue.empty())
            {
                const size_t curr = queue.front();
                queue.pop();
                for (const size_t neighbor : adj[curr])
                {
                    if (component[neighbor] == -1)
                    {
                        component[neighbor] = comp;
                        queue.push(neighbor);
                    }
                }
            }
        }

        std::vector<std::vector<size_t>> components(n_components);
        for (size_t li = 0; li < faces_group.size(); li++)
            components[component[li]].push_back(faces_group[li]);

        for (auto& comp : components)
            if (! comp.empty())
                result.push_back(std::move(comp));
    }

    return result;
}

/*!
 * When loading the mesh, each vertex of each triangle is given a unique index, even if it is used in multiple adjacent triangles. The purpose
 * of this function is to remove double vertices so that we can make adjacency detection easier.
 * @param faces The original list of faces
 * @param vertices The original list of vertices position
 * @return The modified list of faces, which contains as many faces but with merged vertices
 */
std::vector<Face> groupSimilarVertices(const std::vector<Face>& faces, const std::vector<Point3F>& vertices)
{
    std::vector<Face> faces_with_similar_indices;
    std::map<Point3F, size_t> unique_vertices_indices;
    std::vector<uint32_t> new_vertices_indices(vertices.size());

    for (const auto [index, vertex] : vertices | ranges::views::enumerate)
    {
        auto iterator = unique_vertices_indices.find(vertex);
        if (iterator == unique_vertices_indices.end())
        {
            // This is the very first time we see this position, register it
            unique_vertices_indices[vertex] = index;
            new_vertices_indices[index] = index;
        }
        else
        {
            new_vertices_indices[index] = iterator->second;
        }
    }

    for (const Face& face : faces)
    {
        faces_with_similar_indices.push_back(Face{ new_vertices_indices[face.i1], new_vertices_indices[face.i2], new_vertices_indices[face.i3] });
    }

    return faces_with_similar_indices;
}

/*!
 * Detects self-overlapping UV projections within charts and splits them into non-overlapping layers.
 * When a curved surface is projected orthographically, the projection can fold over itself. This function
 * rasterizes each chart's UV triangles onto a grid (barycentric point-in-triangle) to detect such overlap,
 * then uses BFS-based layering to partition each overlapping chart into non-overlapping sub-charts.
 */
static std::vector<std::vector<size_t>> splitSelfOverlappingCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& faces,
    const std::vector<Point2F>& uv_coords)
{
    constexpr int GRID_RES = 128;
    constexpr int GRID_CELLS = GRID_RES * GRID_RES;
    std::vector<std::vector<size_t>> result;

    for (const auto& chart : charts)
    {
        if (chart.size() <= 2)
        {
            result.push_back(chart);
            continue;
        }

        float umin = std::numeric_limits<float>::max(), umax = std::numeric_limits<float>::lowest();
        float vmin = std::numeric_limits<float>::max(), vmax = std::numeric_limits<float>::lowest();
        for (const size_t fi : chart)
        {
            for (const uint32_t vi : { faces[fi].i1, faces[fi].i2, faces[fi].i3 })
            {
                umin = std::min(umin, uv_coords[vi].x);
                umax = std::max(umax, uv_coords[vi].x);
                vmin = std::min(vmin, uv_coords[vi].y);
                vmax = std::max(vmax, uv_coords[vi].y);
            }
        }

        const float du = umax - umin;
        const float dv = vmax - vmin;
        if (du < 1e-10f || dv < 1e-10f)
        {
            result.push_back(chart);
            continue;
        }

        const float scale_u = static_cast<float>(GRID_RES) / du;
        const float scale_v = static_cast<float>(GRID_RES) / dv;

        // Build face adjacency within this chart (faces sharing a vertex are adjacent)
        std::map<uint32_t, std::vector<size_t>> vert_faces;
        for (size_t i = 0; i < chart.size(); i++)
        {
            const Face& f = faces[chart[i]];
            vert_faces[f.i1].push_back(i);
            vert_faces[f.i2].push_back(i);
            vert_faces[f.i3].push_back(i);
        }

        std::vector<std::vector<size_t>> adj(chart.size());
        for (const auto& [v, flist] : vert_faces)
            for (size_t a : flist)
                for (size_t b : flist)
                    if (a != b)
                        adj[a].push_back(b);
        for (auto& a : adj)
        {
            std::sort(a.begin(), a.end());
            a.erase(std::unique(a.begin(), a.end()), a.end());
        }

        // Rasterize a face's UV triangle onto the grid using barycentric point-in-triangle
        auto rasterize = [&](size_t local_idx, std::vector<int>& out)
        {
            out.clear();
            const Face& face = faces[chart[local_idx]];
            const float gx0 = (uv_coords[face.i1].x - umin) * scale_u;
            const float gy0 = (uv_coords[face.i1].y - vmin) * scale_v;
            const float gx1 = (uv_coords[face.i2].x - umin) * scale_u;
            const float gy1 = (uv_coords[face.i2].y - vmin) * scale_v;
            const float gx2 = (uv_coords[face.i3].x - umin) * scale_u;
            const float gy2 = (uv_coords[face.i3].y - vmin) * scale_v;

            const int x0 = std::max(0, static_cast<int>(std::min({ gx0, gx1, gx2 })));
            const int x1 = std::min(GRID_RES - 1, static_cast<int>(std::max({ gx0, gx1, gx2 })));
            const int y0 = std::max(0, static_cast<int>(std::min({ gy0, gy1, gy2 })));
            const int y1 = std::min(GRID_RES - 1, static_cast<int>(std::max({ gy0, gy1, gy2 })));

            const float det = (gx1 - gx0) * (gy2 - gy0) - (gx2 - gx0) * (gy1 - gy0);
            if (std::abs(det) < 1e-10f)
                return;
            const float inv_det = 1.0f / det;

            for (int gy = y0; gy <= y1; gy++)
            {
                const float py = gy + 0.5f;
                for (int gx = x0; gx <= x1; gx++)
                {
                    const float px = gx + 0.5f;
                    const float ba = ((gx1 - px) * (gy2 - py) - (gx2 - px) * (gy1 - py)) * inv_det;
                    const float bb = ((gx2 - px) * (gy0 - py) - (gx0 - px) * (gy2 - py)) * inv_det;
                    const float bc = 1.0f - ba - bb;
                    if (ba > 0.0f && bb > 0.0f && bc > 0.0f)
                        out.push_back(gy * GRID_RES + gx);
                }
            }
        };

        // Quick check: do any non-adjacent faces overlap in UV space?
        std::vector<size_t> detect_grid(GRID_CELLS, SIZE_MAX);
        bool overlap_found = false;
        std::vector<int> cells;

        for (size_t i = 0; i < chart.size() && ! overlap_found; i++)
        {
            rasterize(i, cells);
            for (const int c : cells)
            {
                if (detect_grid[c] != SIZE_MAX && detect_grid[c] != i)
                {
                    if (! std::binary_search(adj[i].begin(), adj[i].end(), detect_grid[c]))
                    {
                        overlap_found = true;
                        break;
                    }
                }
                detect_grid[c] = i;
            }
        }

        if (! overlap_found)
        {
            result.push_back(chart);
            continue;
        }

        // BFS layering: greedily assign faces to non-overlapping layers while preserving connectivity.
        // Each layer gets a grid tracking all occupying faces per cell so that adjacency checks
        // correctly distinguish edge-sharing (harmless) from true overlap.
        std::vector<int> face_layer(chart.size(), -1);
        int layer_count = 0;
        std::vector<std::vector<size_t>> lgrid(GRID_CELLS);

        while (true)
        {
            size_t seed = SIZE_MAX;
            for (size_t i = 0; i < chart.size(); i++)
                if (face_layer[i] == -1)
                {
                    seed = i;
                    break;
                }
            if (seed == SIZE_MAX)
                break;

            const int layer = layer_count++;
            for (auto& cell : lgrid)
                cell.clear();

            std::vector<size_t> queue;
            size_t front = 0;
            queue.push_back(seed);

            while (front < queue.size())
            {
                const size_t curr = queue[front++];
                if (face_layer[curr] != -1)
                    continue;

                rasterize(curr, cells);

                bool conflict = false;
                for (const int c : cells)
                {
                    for (const size_t occ : lgrid[c])
                    {
                        if (occ != curr && ! std::binary_search(adj[curr].begin(), adj[curr].end(), occ))
                        {
                            conflict = true;
                            break;
                        }
                    }
                    if (conflict)
                        break;
                }

                if (conflict)
                    continue;

                face_layer[curr] = layer;
                for (const int c : cells)
                    lgrid[c].push_back(curr);

                for (const size_t neighbor : adj[curr])
                    if (face_layer[neighbor] == -1)
                        queue.push_back(neighbor);
            }
        }

        std::vector<std::vector<size_t>> layers(layer_count);
        for (size_t i = 0; i < chart.size(); i++)
            if (face_layer[i] >= 0)
                layers[face_layer[i]].push_back(chart[i]);

        for (auto& layer : layers)
            if (! layer.empty())
                result.push_back(std::move(layer));
    }

    return result;
}

/*!
 * Splits charts that contain a UV fold — faces whose UV triangles have opposite winding
 * from the chart majority.  On a curved surface projected along a single normal, the fold
 * happens where the surface crosses 90° from that normal; faces beyond the fold get a
 * flipped (negative signed-area) UV triangle.  The grid-based overlap detection misses
 * these because the folded faces are topologically adjacent (share vertices along the fold).
 */
static std::vector<std::vector<size_t>> splitFoldedCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& faces,
    const std::vector<Point2F>& uv_coords)
{
    std::vector<std::vector<size_t>> result;

    for (const auto& chart : charts)
    {
        if (chart.size() <= 2)
        {
            result.push_back(chart);
            continue;
        }

        std::vector<size_t> positive, negative;
        for (const size_t fi : chart)
        {
            const Face& f = faces[fi];
            const Point2F& a = uv_coords[f.i1];
            const Point2F& b = uv_coords[f.i2];
            const Point2F& c = uv_coords[f.i3];

            const float signed_area = (b.x - a.x) * (c.y - a.y)
                                    - (c.x - a.x) * (b.y - a.y);

            if (signed_area >= 0.0f)
                positive.push_back(fi);
            else
                negative.push_back(fi);
        }

        if (negative.empty() || positive.empty())
            result.push_back(chart);
        else
        {
            result.push_back(std::move(positive));
            result.push_back(std::move(negative));
        }
    }

    return result;
}

/*!
 * Splits charts whose UV bounding box has an extreme aspect ratio into shorter segments.
 * Long thin islands (from cylindrical surfaces, pipes, flanges) pack poorly into a square texture.
 * Splitting them into roughly square segments improves packing efficiency.
 */
static std::vector<std::vector<size_t>> splitElongatedCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& faces,
    const std::vector<Point2F>& uv_coords)
{
    constexpr float max_aspect = 8.0f;
    constexpr float target_aspect = 3.0f;
    std::vector<std::vector<size_t>> result;

    for (const auto& chart : charts)
    {
        if (chart.size() <= 4)
        {
            result.push_back(chart);
            continue;
        }

        float umin = std::numeric_limits<float>::max(), umax = std::numeric_limits<float>::lowest();
        float vmin = std::numeric_limits<float>::max(), vmax = std::numeric_limits<float>::lowest();
        for (const size_t fi : chart)
        {
            for (const uint32_t vi : { faces[fi].i1, faces[fi].i2, faces[fi].i3 })
            {
                umin = std::min(umin, uv_coords[vi].x);
                umax = std::max(umax, uv_coords[vi].x);
                vmin = std::min(vmin, uv_coords[vi].y);
                vmax = std::max(vmax, uv_coords[vi].y);
            }
        }

        const float du = umax - umin;
        const float dv = vmax - vmin;
        if (du < 1e-10f || dv < 1e-10f)
        {
            result.push_back(chart);
            continue;
        }

        const float aspect = std::max(du / dv, dv / du);
        if (aspect <= max_aspect)
        {
            result.push_back(chart);
            continue;
        }

        const bool split_along_u = (du > dv);
        const int n_segments = static_cast<int>(std::ceil(aspect / target_aspect));
        const float seg_min = split_along_u ? umin : vmin;
        const float seg_size = (split_along_u ? du : dv) / static_cast<float>(n_segments);

        std::vector<std::vector<size_t>> segments(n_segments);
        for (const size_t fi : chart)
        {
            float centroid = 0.0f;
            for (const uint32_t vi : { faces[fi].i1, faces[fi].i2, faces[fi].i3 })
                centroid += split_along_u ? uv_coords[vi].x : uv_coords[vi].y;
            centroid /= 3.0f;

            const int seg = std::clamp(static_cast<int>((centroid - seg_min) / seg_size), 0, n_segments - 1);
            segments[seg].push_back(fi);
        }

        for (auto& seg : segments)
            if (! seg.empty())
                result.push_back(std::move(seg));
    }

    return result;
}

/*!
 * Merges charts below a minimum face count into the neighboring chart they share the most boundary
 * vertices with, provided their average normals are compatible. Uses union-find for clean cycle handling.
 * @param charts Current face groups
 * @param connectivity_faces Faces with position-merged vertex indices for adjacency detection
 * @param original_faces Original faces for normal computation
 * @param vertices Vertex positions for normal computation
 * @param min_chart_size Charts with fewer faces than this are candidates for merging
 */
static std::vector<std::vector<size_t>> mergeTinyCharts(
    const std::vector<std::vector<size_t>>& charts,
    const std::vector<Face>& connectivity_faces,
    const std::vector<Face>& original_faces,
    const std::vector<Point3F>& vertices,
    size_t min_chart_size)
{
    if (charts.size() <= 1)
        return charts;

    constexpr float min_normal_alignment = 0.707f; // cos(45°)
    std::vector<Vector3F> chart_normals(charts.size());
    for (size_t ci = 0; ci < charts.size(); ci++)
    {
        Vector3F normal;
        for (const size_t fi : charts[ci])
        {
            const Face& face = original_faces[fi];
            const auto tri_normal = geometry_utils::triangleNormal(vertices[face.i1], vertices[face.i2], vertices[face.i3]);
            if (tri_normal.has_value())
                normal += tri_normal.value();
        }
        normal.normalize();
        chart_normals[ci] = normal;
    }

    std::vector<size_t> face_chart(connectivity_faces.size(), SIZE_MAX);
    for (size_t ci = 0; ci < charts.size(); ci++)
        for (const size_t fi : charts[ci])
            face_chart[fi] = ci;

    std::map<uint32_t, std::vector<size_t>> vertex_faces;
    for (size_t fi = 0; fi < connectivity_faces.size(); fi++)
    {
        vertex_faces[connectivity_faces[fi].i1].push_back(fi);
        vertex_faces[connectivity_faces[fi].i2].push_back(fi);
        vertex_faces[connectivity_faces[fi].i3].push_back(fi);
    }

    std::vector<size_t> parent(charts.size());
    std::iota(parent.begin(), parent.end(), size_t(0));
    std::vector<size_t> group_size(charts.size());
    for (size_t ci = 0; ci < charts.size(); ci++)
        group_size[ci] = charts[ci].size();

    auto find = [&](size_t x) -> size_t
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](size_t a, size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        if (group_size[a] < group_size[b])
            std::swap(a, b);
        parent[b] = a;
        group_size[a] += group_size[b];
    };

    for (size_t ci = 0; ci < charts.size(); ci++)
    {
        if (charts[ci].size() >= min_chart_size)
            continue;

        std::map<size_t, size_t> neighbor_shared;
        for (const size_t fi : charts[ci])
        {
            for (const uint32_t vi : { connectivity_faces[fi].i1, connectivity_faces[fi].i2, connectivity_faces[fi].i3 })
            {
                for (const size_t nfi : vertex_faces[vi])
                {
                    const size_t nci = face_chart[nfi];
                    if (nci != SIZE_MAX && find(nci) != find(ci))
                        neighbor_shared[find(nci)]++;
                }
            }
        }

        if (neighbor_shared.empty())
            continue;

        size_t best_neighbor = find(ci);
        size_t best_count = 0;
        for (const auto& [nci, count] : neighbor_shared)
        {
            if (chart_normals[ci].dot(chart_normals[nci]) < min_normal_alignment)
                continue;

            if (count > best_count)
            {
                best_count = count;
                best_neighbor = nci;
            }
        }

        if (best_neighbor != find(ci))
            unite(ci, best_neighbor);
    }

    std::map<size_t, std::vector<size_t>> merged;
    for (size_t ci = 0; ci < charts.size(); ci++)
    {
        const size_t root = find(ci);
        for (const size_t fi : charts[ci])
            merged[root].push_back(fi);
    }

    std::vector<std::vector<size_t>> result;
    result.reserve(merged.size());
    for (auto& [root, chart_faces] : merged)
        result.push_back(std::move(chart_faces));

    return result;
}

/*!
 * Duplicates vertices at chart boundaries so each chart exclusively owns its vertices,
 * then re-projects each chart's vertices along the chart's average normal.
 */
static void duplicateAndReproject(
    std::vector<Point3F>& vertices,
    std::vector<Face>& faces,
    std::vector<Point2F>& uv_coords,
    const std::vector<Face>& original_faces,
    const std::vector<std::vector<size_t>>& charts,
    std::vector<uint32_t>* out_vertex_remap = nullptr)
{
    std::vector<uint32_t> vertex_owner(vertices.size(), UINT32_MAX);
    std::vector<uint32_t> vertex_remap(vertices.size());
    std::iota(vertex_remap.begin(), vertex_remap.end(), 0u);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> dup_map;

    std::vector<bool> face_in_chart(faces.size(), false);

    for (uint32_t chart_idx = 0; chart_idx < static_cast<uint32_t>(charts.size()); ++chart_idx)
    {
        for (const size_t face_idx : charts[chart_idx])
        {
            face_in_chart[face_idx] = true;
            Face& face = faces[face_idx];
            for (uint32_t* vi : { &face.i1, &face.i2, &face.i3 })
            {
                if (vertex_owner[*vi] == chart_idx)
                    continue;

                if (vertex_owner[*vi] == UINT32_MAX)
                {
                    vertex_owner[*vi] = chart_idx;
                }
                else
                {
                    auto key = std::make_pair(*vi, chart_idx);
                    auto it = dup_map.find(key);
                    if (it != dup_map.end())
                    {
                        *vi = it->second;
                    }
                    else
                    {
                        const auto new_idx = static_cast<uint32_t>(uv_coords.size());
                        uv_coords.push_back(uv_coords[*vi]);
                        vertices.push_back(vertices[*vi]);
                        vertex_owner.push_back(chart_idx);
                        vertex_remap.push_back(*vi);
                        dup_map[key] = new_idx;
                        *vi = new_idx;
                    }
                }
            }
        }
    }

    // Detach vertices of uncharted faces (degenerate triangles) from charted ones.
    // Without this, degenerate faces share vertices with charted faces, creating
    // long thin triangles spanning between packed UV islands or from the UV origin.
    for (size_t fi = 0; fi < faces.size(); fi++)
    {
        if (face_in_chart[fi])
            continue;
        Face& face = faces[fi];
        for (uint32_t* vi : { &face.i1, &face.i2, &face.i3 })
        {
            if (vertex_owner[*vi] != UINT32_MAX)
            {
                const auto new_idx = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertices[*vi]);
                uv_coords.push_back(Point2F{ .x = 0.0f, .y = 0.0f });
                vertex_owner.push_back(UINT32_MAX);
                vertex_remap.push_back(vertex_remap[*vi]);
                *vi = new_idx;
            }
        }
    }

    for (uint32_t chart_idx = 0; chart_idx < static_cast<uint32_t>(charts.size()); ++chart_idx)
    {
        Vector3F avg_normal;
        for (const size_t face_idx : charts[chart_idx])
        {
            const Face& orig_face = original_faces[face_idx];
            const auto tri_normal = geometry_utils::triangleNormal(
                vertices[orig_face.i1], vertices[orig_face.i2], vertices[orig_face.i3]);
            if (tri_normal.has_value())
                avg_normal += tri_normal.value();
        }
        if (! avg_normal.normalize())
            continue;

        const Matrix33F axis_mat = Matrix33F::makeOrthogonalBasis(avg_normal);

        for (const size_t face_idx : charts[chart_idx])
        {
            const Face& face = faces[face_idx];
            for (const uint32_t vi : { face.i1, face.i2, face.i3 })
            {
                uv_coords[vi] = axis_mat.project(vertices[vertex_remap[vi]]);
            }
        }

    }

    if (out_vertex_remap)
        *out_vertex_remap = std::move(vertex_remap);
}

static void restorePreDuplicationState(
    std::vector<Point3F>& vertices,
    std::vector<Point2F>& uv_coords,
    std::vector<Face>& faces,
    const std::vector<Face>& original_faces,
    size_t original_vertex_count)
{
    vertices.erase(vertices.begin() + static_cast<std::ptrdiff_t>(original_vertex_count), vertices.end());
    uv_coords.erase(uv_coords.begin() + static_cast<std::ptrdiff_t>(original_vertex_count), uv_coords.end());
    faces = original_faces;
}

/*!
 * Packs the charts (faces groups) onto a texture image by using as much space as possible without having them overlap
 * @param vertices The list of vertices position
 * @param faces The list of vertices position
 * @param charts The list of grouped faces indices
 * @param uv_coords The original UV coordinates, which may be overlapping and not fitting on an image. As an output they
 *                  will be properly scaled and distributed on the image.
 * @param texture_width Output width to be used for the texture image
 * @param texture_height Output height to be used for the texture image
 * @return
 */
bool packCharts(
    const std::vector<Point3F>& vertices,
    const std::vector<Face>& faces,
    const std::vector<std::vector<size_t>>& charts,
    std::vector<Point2F>& uv_coords,
    uint32_t& texture_width,
    uint32_t& texture_height)
{
    xatlas::Atlas* atlas = xatlas::Create();
    xatlas::UvMeshDecl mesh;
    mesh.vertexUvData = uv_coords.data();
    mesh.indexData = faces.data();
    mesh.vertexCount = static_cast<uint32_t>(uv_coords.size());
    mesh.vertexStride = sizeof(Point2F);
    mesh.indexCount = faces.size() * 3;
    mesh.indexFormat = xatlas::IndexFormat::UInt32;

    if (xatlas::AddUvMesh(atlas, mesh) != xatlas::AddMeshError::Success)
    {
        xatlas::Destroy(atlas);
        spdlog::error("Error adding mesh");
        return false;
    }

    constexpr uint32_t calculation_definition = 512;
    constexpr uint32_t desired_definition = 4096;

    xatlas::SetCharts(atlas, charts);

    constexpr xatlas::PackOptions pack_options{ .padding = 0, .resolution = calculation_definition };
    xatlas::PackCharts(atlas, pack_options);

    texture_width = atlas->width;
    texture_height = atlas->height;
    const uint32_t max_side = std::max(texture_width, texture_height);
    const double scale = static_cast<double>(desired_definition) / static_cast<double>(max_side);
    texture_width = std::llrint(texture_width * scale);
    texture_height = std::llrint(texture_height * scale);

    const xatlas::Mesh& output_mesh = *atlas->meshes;
    const auto width = static_cast<float>(atlas->width);
    const auto height = static_cast<float>(atlas->height);
    for (size_t i = 0; i < output_mesh.vertexCount; ++i)
    {
        const xatlas::PlacedVertex& vertex = output_mesh.vertexArray[i];
        uv_coords[vertex.xref] = Point2F{ .x = vertex.uv[0] / width, .y = vertex.uv[1] / height };
    }

    xatlas::Destroy(atlas);
    return true;
}

bool smartUnwrap(std::vector<Point3F>& vertices, std::vector<Face>& faces, std::vector<Point2F>& uv_coords,
                 uint32_t& texture_width, uint32_t& texture_height, std::vector<uint32_t>& vertex_remap)
{
    const std::vector<Face> original_faces(faces);
    std::vector<std::vector<size_t>> charts = makeCharts(vertices, faces, uv_coords);

    const std::vector<Face> faces_with_similar_indices = groupSimilarVertices(faces, vertices);
    charts = splitNonLinkedFacesCharts(charts, faces_with_similar_indices);

    const size_t original_vertex_count = vertices.size();

    // Phase 1: Detect and split overlapping / elongated / folded charts
    duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts);
    charts = splitSelfOverlappingCharts(charts, faces, uv_coords);
    charts = splitFoldedCharts(charts, faces, uv_coords);
    charts = splitElongatedCharts(charts, faces, uv_coords);
    restorePreDuplicationState(vertices, uv_coords, faces, original_faces, original_vertex_count);

    // Iterative merge-split: each iteration merges aggressively then runs overlap
    // detection so that folded charts are split back.  Fragments from one iteration
    // become candidates for absorption into *different* neighbors in the next.
    constexpr size_t merge_iters = 3;
    const size_t merge_thresholds[merge_iters] = {
        std::max<size_t>(original_faces.size() / 500, 32),
        std::max<size_t>(original_faces.size() / 2000, 16),
        std::max<size_t>(original_faces.size() / 8000, 4),
    };

    for (size_t iter = 0; iter < merge_iters; ++iter)
    {
        charts = mergeTinyCharts(charts, faces_with_similar_indices, original_faces, vertices, merge_thresholds[iter]);

        duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts);
        charts = splitSelfOverlappingCharts(charts, faces, uv_coords);
        charts = splitFoldedCharts(charts, faces, uv_coords);
        restorePreDuplicationState(vertices, uv_coords, faces, original_faces, original_vertex_count);
    }

    // Phase 4: Final vertex duplication and re-projection
    duplicateAndReproject(vertices, faces, uv_coords, original_faces, charts, &vertex_remap);

    if (! packCharts(vertices, faces, charts, uv_coords, texture_width, texture_height))
        return false;

    // Strip degenerate faces (all 3 vertices at UV origin) and compact the vertex array.
    {
        const auto at_origin = [](const Point2F& p) { return p.x == 0.0f && p.y == 0.0f; };

        std::vector<Face> clean_faces;
        clean_faces.reserve(faces.size());
        for (const Face& f : faces)
        {
            if (at_origin(uv_coords[f.i1]) && at_origin(uv_coords[f.i2]) && at_origin(uv_coords[f.i3]))
                continue;
            clean_faces.push_back(f);
        }

        std::vector<uint32_t> idx_remap(vertices.size(), UINT32_MAX);
        std::vector<Point3F> compact_verts;
        std::vector<Point2F> compact_uvs;
        std::vector<uint32_t> compact_vremap;
        compact_verts.reserve(clean_faces.size());
        compact_uvs.reserve(clean_faces.size());
        compact_vremap.reserve(clean_faces.size());

        for (Face& f : clean_faces)
        {
            for (uint32_t* vi : { &f.i1, &f.i2, &f.i3 })
            {
                if (idx_remap[*vi] == UINT32_MAX)
                {
                    idx_remap[*vi] = static_cast<uint32_t>(compact_verts.size());
                    compact_verts.push_back(vertices[*vi]);
                    compact_uvs.push_back(uv_coords[*vi]);
                    compact_vremap.push_back(vertex_remap[*vi]);
                }
                *vi = idx_remap[*vi];
            }
        }

        vertices = std::move(compact_verts);
        uv_coords = std::move(compact_uvs);
        vertex_remap = std::move(compact_vremap);
        faces = std::move(clean_faces);
    }

    return true;
}
