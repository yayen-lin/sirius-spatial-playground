#include <cstdint>

namespace sirius_geometry 
{ 
  enum class GeometryType: uint8_t{
    POINT = 0, 
    LINESTRING, 
    POLYGON, 
    MULTIPOINT, 
    MULTILINESTRING
  };

  struct GeometryTypes{
    __host__ static bool isSinglePart(GeometryType type){
      return type == GeometryType::POINT || type == GeometryType::LINESTRING || type == GeometryType::POLYGON;
    }

    __host__ __device__ static bool isMuliPart(GeometryType type){ 
      return type == GeometryType::POLYGON || type ==  GeometryType::MULTIPOINT || type == GeometryType::MULTILINESTRING;
    }
  };


  enum class SerializedGeometryType: uint32_t { 
    POINT = 0, 
    LINESTRING, 
    POLYGON, 
    MULTIPOINT, 
    MULTILINESTRING
  };

  // A bounding box struct
  template <typename T> 
  struct Box2D { 
    struct Vec2 {T x, y;};
    Vec2 min, max;
  };

  struct GeometryProperties { 
    __host__ __device__ void CheckVersion() const {}
    __host__ __device__ bool HasBBox() const { return false; }
  };

  template <typename T>
  __host__ __device__ T Load(const char* ptr) {
      return *reinterpret_cast<const T*>(ptr);
  }
  
  class geometry_t{ 
    private: 
      const char* data; 
    public:
      __host__ __device__ geometry_t(): data(nullptr){}
      __host__ __device__ explicit geometry_t(const char* d) : data(d) {}
      __host__ __device__ GeometryType GetType() const {
        return Load<GeometryType>(data);
      }
      __host__ __device__ GeometryProperties GetProperties() const {
          return Load<GeometryProperties>(data + 1);
      }
      __host__ __device__ bool TryGetCachedBounds(Box2D<float>& bbox) const{ 
        if(!data) return false; 
        GeometryType header_type = Load<GeometryType>(data); 
        GeometryProperties properties = Load<GeometryProperties>(data + 1); 
        properties.CheckVersion();

        if(properties.HasBBox()){ 
          const float *fptr = reinterpret_cast<const float*>(data + 1 + sizeof(GeometryProperties) + 4); 
          bbox.min.x = fptr[0];
          bbox.min.y = fptr[1]; 
          bbox.max.x = fptr[2]; 
          bbox.max.y = fptr[3]; 

        }
        if(header_type == GeometryType::POINT){ 
          const double* dptr = reinterpret_cast<const double*>(data + 1 + sizeof(GeometryProperties) + 4 + 8);
          bbox.min.x = static_cast<float>(dptr[0]); 
          bbox.min.y = static_cast<float>(dptr[1]); 
          bbox.max.x = static_cast<float>(dptr[0]); 
          bbox.max.y = static_cast<float>(dptr[1]); 
          return true;
        }
      }
  };
}