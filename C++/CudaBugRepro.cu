// compile with /usr/local/cuda/bin/nvcc -std=c++11 CudaBugRepro.cu

namespace at {
  namespace detail {

    template <typename T>
    struct Info {
    };

  }

  template <typename scalar1>
  __global__ void
  kernelPointwiseApply2a(detail::Info<scalar1> a) {
  }

  template <typename scalar1>
  void tensor_apply2a() {
    return kernelPointwiseApply2a<scalar1>
      <<<0, 0, 0, nullptr>>>
         (detail::Info<scalar1>());
  }

  namespace detail {

  }

}

namespace detail {

template <typename T>
struct CType;

template<>
struct CType<float> {
  using type = float;
  static constexpr float t = 0;
};
}

using namespace at;

int main() {
  //using scalar_t = ::detail::CType<float>::type; //BROKEN
  using scalar_t = decltype(::detail::CType<float>::t);
  at::tensor_apply2a<scalar_t>();

  return 0;
}
