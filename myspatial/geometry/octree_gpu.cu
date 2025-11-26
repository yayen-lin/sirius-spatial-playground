// octree_gpu.cu
// Simple GPU octree: parallel insertion + radius queries
// Compile: nvcc -O3 octree_gpu.cu -o octree_gpu

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <assert.h>

#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

struct float3_ {
    float x,y,z;
};
__host__ __device__ inline float3_ make_float3_(float x,float y,float z){ float3_ f={x,y,z}; return f; }

struct OctreeParams {
    float3_ origin;   // center of root bounding cube
    float halfSize;   // half-size of root cube (cube extends origin +/- halfSize)
    int maxDepth;
};

// Node representation stored on device
struct OctNode {
    int childs[8];   // indices of children, -1 if none
    int head;        // head of linked list of point indices for this node (leaf only), -1 if none
    // We purposely do NOT store bounds per node to save memory; bounds are computed during traversal.
};

__device__ inline int childIndexFromPos(bool bx,bool by,bool bz){
    return (bx<<2) | (by<<1) | (bz<<0);
}

// Device arrays (pointers will be allocated by host and passed to kernels)
struct OctreeDevice {
    OctNode* nodes;   // array of nodes [nodePoolSize]
    int* nodeCounter; // single int in device memory used with atomicAdd to allocate nodes
    int nodePoolSize;

    int* pointNext;   // per point next pointer (for leaf linked lists), length = numPoints
    int* pointIndices; // optional alias, not necessary

    float3_* points;  // point coordinates
    int numPoints;

    OctreeParams params;
};

// Initialize node pool (on device)
__global__ void initNodePool(OctNode* nodes, int nodePoolSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < nodePoolSize; i += blockDim.x * gridDim.x) {
        for (int c=0;c<8;++c) nodes[i].childs[c] = -1;
        nodes[i].head = -1;
    }
}

// Kernel: insert points in parallel
__global__ void insertPointsKernel(OctNode* nodes, int* nodeCounter, int nodePoolSize,
                                   int* pointNext, const float3_* points, int numPoints,
                                   OctreeParams params)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPoints) return;

    float3_ p = points[tid];

    // Start at root, index 0 is root (we assume host allocated and reserved it)
    int nodeIdx = 0;
    float3_ origin = params.origin;
    float half = params.halfSize;

    // Traverse until leaf (depth maxDepth)
    for (int depth = 0; depth < params.maxDepth; ++depth) {
        // compute which child the point goes into
        bool bx = p.x >= origin.x;
        bool by = p.y >= origin.y;
        bool bz = p.z >= origin.z;
        int c = childIndexFromPos(bx, by, bz);

        // Compute child's center for next iteration
        float childHalf = half * 0.5f;
        float3_ childCenter = origin;
        childCenter.x += (bx ? childHalf : -childHalf);
        childCenter.y += (by ? childHalf : -childHalf);
        childCenter.z += (bz ? childHalf : -childHalf);

        // Try to allocate child if missing using atomicCAS
        int* childPtr = &nodes[nodeIdx].childs[c];
        int cur = atomicCAS(childPtr, -1, -2); // -2 means "being created" marker
        if (cur == -1) {
            // we reserved slot, now allocate a node from pool
            int newNode = atomicAdd(nodeCounter, 1);
            if (newNode >= nodePoolSize) {
                // Out of nodes - roll back pointer to -1 and stop (silently drop)
                // set child's pointer back to -1
                atomicExch(childPtr, -1);
                // fallback: attach point to current node (treat as leaf)
                break;
            }
            // initialize child (we rely on init kernel or threads to set fields; but since allocated dynamically, set here)
            for (int i=0;i<8;++i) nodes[newNode].childs[i] = -1;
            nodes[newNode].head = -1;
            // store newNode into childPtr (it is currently -2)
            atomicExch(childPtr, newNode);
            cur = newNode;
        } else if (cur == -2) {
            // some other thread is creating node; spin/wait until it's created
            // busy-wait is OK for short times; but must be careful to avoid infinite loop
            int val;
            do {
                val = *childPtr;
            } while (val == -2);
            cur = val;
            if (cur == -1) {
                // some error fallback; treat as leaf
                break;
            }
        } else {
            // child exists (cur >= 0)
        }

        // advance down tree
        nodeIdx = cur;
        origin = childCenter;
        half = childHalf;
    }

    // Insert this point at nodeIdx's head using lock-free singly linked list:
    // pointNext[tid] = oldHead; head = tid (atomicExchange)
    int oldHead = atomicExch(&nodes[nodeIdx].head, tid);
    pointNext[tid] = oldHead;
}

// Radius query kernel: for each query point, count points inside radius^2
__global__ void radiusQueryKernel(const OctNode* __restrict__ nodes,
                                  const int* __restrict__ pointNext,
                                  const float3_* __restrict__ points,
                                  int nodePoolSize,
                                  OctreeParams params,
                                  const float3_* __restrict__ queryPts,
                                  int numQueries,
                                  float radius,
                                  int* outCounts)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= numQueries) return;
    float r2 = radius * radius;
    float3_ q = queryPts[qid];

    // We'll do a simple stack-based traversal on device to visit nodes whose bounds intersect the sphere
    // Stack size = maxDepth+1 small (we use 64 as safe upper bound)
    const int MAX_STACK = 64;
    int stackNode[MAX_STACK];
    float3_ stackOrigin[MAX_STACK];
    float stackHalf[MAX_STACK];
    int sp = 0;

    // push root
    stackNode[sp] = 0;
    stackOrigin[sp] = params.origin;
    stackHalf[sp] = params.halfSize;
    sp++;

    int count = 0;

    while (sp > 0) {
        sp--;
        int nodeIdx = stackNode[sp];
        float3_ origin = stackOrigin[sp];
        float half = stackHalf[sp];

        // Check cube-sphere intersection (AABB-sphere)
        // Compute squared distance from sphere center to AABB
        float dx = fmaxf(fabsf(q.x - origin.x) - half, 0.0f);
        float dy = fmaxf(fabsf(q.y - origin.y) - half, 0.0f);
        float dz = fmaxf(fabsf(q.z - origin.z) - half, 0.0f);
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > r2) continue; // no intersection

        // If leaf (no children), traverse its point list
        bool hasChild = false;
        for (int c=0;c<8;++c) {
            int ch = nodes[nodeIdx].childs[c];
            if (ch >= 0) {
                hasChild = true;
                // push child
                if (sp+1 >= MAX_STACK) continue; // stack overflow guard
                float3_ childCenter = origin;
                float childHalf = half * 0.5f;
                // determine child center offset by bits corresponding to c
                bool bx = (c & 4) != 0;
                bool by = (c & 2) != 0;
                bool bz = (c & 1) != 0;
                childCenter.x += (bx ? childHalf : -childHalf);
                childCenter.y += (by ? childHalf : -childHalf);
                childCenter.z += (bz ? childHalf : -childHalf);
                stackNode[sp] = ch;
                stackOrigin[sp] = childCenter;
                stackHalf[sp] = childHalf;
                sp++;
            }
        }

        if (!hasChild) {
            // leaf: iterate points in linked list
            int pid = nodes[nodeIdx].head;
            while (pid != -1) {
                float3_ pp = points[pid];
                float dxp = pp.x - q.x;
                float dyp = pp.y - q.y;
                float dzp = pp.z - q.z;
                float dist2 = dxp*dxp + dyp*dyp + dzp*dzp;
                if (dist2 <= r2) ++count;
                pid = pointNext[pid];
            }
        } else {
            // Also count points that may be stored in this internal node's head (we allow points at any internal node)
            int pid = nodes[nodeIdx].head;
            while (pid != -1) {
                float3_ pp = points[pid];
                float dxp = pp.x - q.x;
                float dyp = pp.y - q.y;
                float dzp = pp.z - q.z;
                float dist2 = dxp*dxp + dyp*dyp + dzp*dzp;
                if (dist2 <= r2) ++count;
                pid = pointNext[pid];
            }
        }
    }

    outCounts[qid] = count;
}

// Host helper: allocate device structures and run kernels
int main() {
    printf("GPU Octree demo\n");

    // Parameters
    const int N = 1<<20; // number of points (~1M)
    const int Q = 1024;  // number of query points
    const int maxDepth = 8; // depth -> leaf cell size = rootSize/(2^maxDepth)
    const float rootHalfSize = 1.0f; // root cube is [-1,1]^3 if origin at 0
    const float radius = 0.02f;

    // generate random points in [-1,1]^3
    std::vector<float3_> h_points(N);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i=0;i<N;++i) {
        h_points[i] = make_float3_(dist(rng), dist(rng), dist(rng));
    }
    std::vector<float3_> h_queries(Q);
    for (int i=0;i<Q;++i) {
        h_queries[i] = make_float3_(dist(rng), dist(rng), dist(rng));
    }

    // Device allocations
    float3_* d_points = nullptr;
    float3_* d_queries = nullptr;
    int* d_pointNext = nullptr;
    OctNode* d_nodes = nullptr;
    int* d_nodeCounter = nullptr;
    int* d_outCounts = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(float3_)*N));
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), sizeof(float3_)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_queries, sizeof(float3_)*Q));
    CUDA_CHECK(cudaMemcpy(d_queries, h_queries.data(), sizeof(float3_)*Q, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_pointNext, sizeof(int)*N));
    CUDA_CHECK(cudaMemset(d_pointNext, -1, sizeof(int)*N)); // initialize

    // estimate nodePoolSize: worst-case nodes = 1 + 8 + 8^2 + ... + 8^maxDepth = (8^(d+1)-1)/7
    // but that's huge; pick a reasonable pool size, e.g., min(N, something)
    // We'll pick nodePoolSize = min( (8^(maxDepth+1)-1)/7, N )
    size_t poolEstimate = 1;
    size_t accum = 0;
    for (int d=0; d<=maxDepth; ++d) {
        accum += poolEstimate;
        poolEstimate *= 8;
    }
    size_t nodePoolSize = (size_t)accum;
    if (nodePoolSize > (size_t)N) nodePoolSize = N; // cap to N
    if (nodePoolSize < 1024) nodePoolSize = 1024; // minimum
    printf("Node pool size: %zu\n", nodePoolSize);

    CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(OctNode)*nodePoolSize));
    CUDA_CHECK(cudaMalloc(&d_nodeCounter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nodeCounter, 0, sizeof(int)));

    // initialize nodes
    int threads = 256;
    int blocks = (int)((nodePoolSize + threads - 1)/threads);
    initNodePool<<<blocks, threads>>>(d_nodes, (int)nodePoolSize);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // We will reserve node 0 as root now: allocate nodeCounter to 1
    int one = 1;
    CUDA_CHECK(cudaMemcpy(d_nodeCounter, &one, sizeof(int), cudaMemcpyHostToDevice));

    // Prepare octree params
    OctreeParams paramsHost;
    paramsHost.origin = make_float3_(0.0f, 0.0f, 0.0f);
    paramsHost.halfSize = rootHalfSize;
    paramsHost.maxDepth = maxDepth;

    // Insert points kernel
    int blockSize = 256;
    int nBlocks = (N + blockSize - 1) / blockSize;
    // Note: kernel uses d_nodes, d_nodeCounter, d_pointNext, d_points
    // For safety, set root node's fields to initialized values (host->device copy)
    OctNode rootInit;
    for (int i=0;i<8;++i) rootInit.childs[i] = -1;
    rootInit.head = -1;
    CUDA_CHECK(cudaMemcpy(d_nodes, &rootInit, sizeof(OctNode), cudaMemcpyHostToDevice));

    // Launch insertion
    printf("Inserting %d points ...\n", N);
    auto t0 = std::chrono::high_resolution_clock::now();
    insertPointsKernel<<<nBlocks, blockSize>>>(d_nodes, d_nodeCounter, (int)nodePoolSize,
                                               d_pointNext, d_points, N, paramsHost);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    printf("Insertion kernel completed in %.3f ms\n", ms);

    // Query kernel: count points within radius for each query point
    CUDA_CHECK(cudaMalloc(&d_outCounts, sizeof(int)*Q));
    int qBlocks = (Q + blockSize - 1) / blockSize;
    printf("Running %d queries ...\n", Q);
    t0 = std::chrono::high_resolution_clock::now();
    radiusQueryKernel<<<qBlocks, blockSize>>>(d_nodes, d_pointNext, d_points, (int)nodePoolSize, paramsHost,
                                              d_queries, Q, radius, d_outCounts);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    printf("Query kernel completed in %.3f ms\n", ms);

    // Copy back some results
    std::vector<int> h_counts(Q);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_outCounts, sizeof(int)*Q, cudaMemcpyDeviceToHost));
    long long total = 0;
    for (int i=0;i<Q;++i) total += h_counts[i];
    printf("Total hits across queries: %lld\n", total);

    // cleanup
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_pointNext));
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_nodeCounter));
    CUDA_CHECK(cudaFree(d_outCounts));

    printf("Done.\n");
    return 0;
}
