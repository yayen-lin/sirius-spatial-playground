#include "geometry_type.cu"

using namespace sirius_geometry;

__global__ void TestGeometryKernel(const char* geom_data,
                                   int geom_stride,
                                   int count,
                                   float* out_minx,
                                   float* out_miny,
                                   float* out_maxx,
                                   float* out_maxy,
                                   int*   out_success)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const char* ptr = geom_data + tid * geom_stride;

    sirius_geometry::geometry_t g(ptr);

    sirius_geometry::Box2D<float> bbox;
    bool ok = g.TryGetCachedBounds(bbox);

    out_success[tid] = ok ? 1 : 0;

    if (ok) {
        out_minx[tid] = bbox.min.x;
        out_miny[tid] = bbox.min.y;
        out_maxx[tid] = bbox.max.x;
        out_maxy[tid] = bbox.max.y;
    }
}
