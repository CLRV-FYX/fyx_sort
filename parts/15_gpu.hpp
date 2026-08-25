// ===========================================================================
//  Section 15 -- GPU layer (skeleton + CPU fallback)
//
//  SCOPE / HONESTY NOTE: this sandbox has no GPU and no network, so the GPU
//  compute path CANNOT be compiled or run here.  Per DESIGN.md section 16 the
//  GPU layer's agreed scope is "skeleton + CPU fallback", so this file:
//    * probes for a CUDA driver at runtime via dlopen (no GPU headers needed
//      to compile -- symbols are resolved through dlsym);
//    * exposes a device-buffer / stream abstraction;
//    * routes fyx::sort through gpu_sort_dispatch, which returns false on any
//      failure so the caller falls back to the verified CPU kernels.
//  The actual device kernel is opt-in behind FYX_GPU_COMPUTE (off by default)
//  because it is UNVERIFIED without a GPU; enabling it is for GPU boxes where
//  the kernel can be debugged against real hardware.
//
//  Build: define FYX_ENABLE_GPU to include this file.  Default build skips it.
// ===========================================================================

#if FYX_ENABLE_GPU
#include <dlfcn.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fyx {
namespace detail {

// Opaque CUDA driver / NVRTC types (we do not include cuda.h; approximate
// definitions are enough for passing pointers through dlsym).
typedef int                 CUdevice;
typedef void*               CUcontext;
typedef void*               CUmodule;
typedef void*               CUfunction;
typedef void*               CUstream;
typedef unsigned long long  CUdeviceptr;
typedef void*               nvrtcProgram;

// ---- resolved driver symbols ----------------------------------------------
struct CudaSyms {
    void* lib = nullptr;
    void* nvrtc = nullptr;
    // driver API
    int (*cuInit)(unsigned int) = nullptr;
    int (*cuDeviceGet)(CUdevice*, int) = nullptr;
    int (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
    int (*cuMemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    int (*cuMemcpyHtoD)(CUdeviceptr, const void*, std::size_t) = nullptr;
    int (*cuMemcpyDtoH)(void*, CUdeviceptr, std::size_t) = nullptr;
    int (*cuModuleLoadData)(CUmodule*, const char*) = nullptr;
    int (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    int (*cuLaunchKernel)(CUfunction, unsigned,unsigned,unsigned,
                          unsigned,unsigned,unsigned,
                          unsigned, CUstream, void**, void**) = nullptr;
    int (*cuMemFree)(CUdeviceptr) = nullptr;
    int (*cuCtxDestroy)(CUcontext) = nullptr;
    // NVRTC
    int (*nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*,
                              int, const char**, const char**) = nullptr;
    int (*nvrtcCompileProgram)(nvrtcProgram, int, const char**) = nullptr;
    int (*nvrtcGetPTX)(nvrtcProgram, char*) = nullptr;
    int (*nvrtcDestroyProgram)(nvrtcProgram*) = nullptr;

    bool ok = false;
};

inline CudaSyms& cuda_syms() {
    static CudaSyms s;
    if (s.ok) return s;
    // dlopen the driver + compiler; if either is missing we simply stay disabled.
    s.lib   = dlopen("libcuda.so",      RTLD_LAZY | RTLD_LOCAL);
    s.nvrtc = dlopen("libnvrtc.so",     RTLD_LAZY | RTLD_LOCAL);
    if (!s.lib) { if (s.nvrtc) dlclose(s.nvrtc); return s; }
    auto sym = [](void* h, const char* n) -> void* {
        return h ? dlsym(h, n) : nullptr;
    };
    s.cuInit              = (decltype(s.cuInit))             sym(s.lib, "cuInit");
    s.cuDeviceGet         = (decltype(s.cuDeviceGet))        sym(s.lib, "cuDeviceGet");
    s.cuCtxCreate         = (decltype(s.cuCtxCreate))        sym(s.lib, "cuCtxCreate");
    s.cuMemAlloc          = (decltype(s.cuMemAlloc))          sym(s.lib, "cuMemAlloc");
    s.cuMemcpyHtoD        = (decltype(s.cuMemcpyHtoD))        sym(s.lib, "cuMemcpyHtoD");
    s.cuMemcpyDtoH        = (decltype(s.cuMemcpyDtoH))        sym(s.lib, "cuMemcpyDtoH");
    s.cuModuleLoadData    = (decltype(s.cuModuleLoadData))    sym(s.lib, "cuModuleLoadData");
    s.cuModuleGetFunction = (decltype(s.cuModuleGetFunction)) sym(s.lib, "cuModuleGetFunction");
    s.cuLaunchKernel      = (decltype(s.cuLaunchKernel))      sym(s.lib, "cuLaunchKernel");
    s.cuMemFree           = (decltype(s.cuMemFree))           sym(s.lib, "cuMemFree");
    s.cuCtxDestroy        = (decltype(s.cuCtxDestroy))        sym(s.lib, "cuCtxDestroy");
    if (s.nvrtc) {
        s.nvrtcCreateProgram   = (decltype(s.nvrtcCreateProgram))   sym(s.nvrtc, "nvrtcCreateProgram");
        s.nvrtcCompileProgram  = (decltype(s.nvrtcCompileProgram))  sym(s.nvrtc, "nvrtcCompileProgram");
        s.nvrtcGetPTX          = (decltype(s.nvrtcGetPTX))          sym(s.nvrtc, "nvrtcGetPTX");
        s.nvrtcDestroyProgram  = (decltype(s.nvrtcDestroyProgram))  sym(s.nvrtc, "nvrtcDestroyProgram");
    }
    s.ok = s.cuInit && s.cuDeviceGet && s.cuCtxCreate && s.cuMemAlloc &&
           s.cuMemcpyHtoD && s.cuMemcpyDtoH && s.cuModuleLoadData &&
           s.cuModuleGetFunction && s.cuLaunchKernel && s.cuMemFree && s.cuCtxDestroy &&
           s.nvrtcCreateProgram && s.nvrtcCompileProgram && s.nvrtcGetPTX && s.nvrtcDestroyProgram;
    return s;
}

// ---- device buffer ---------------------------------------------------------
template <class T>
struct GpuBuffer {
    CudaSyms*   s = nullptr;
    CUdeviceptr  dev = 0;
    std::size_t  n = 0;
    ~GpuBuffer() { if (s && dev && s->cuMemFree) s->cuMemFree(dev); }
    bool alloc(CudaSyms& syms, std::size_t count) {
        s = &syms; n = count;
        return syms.cuMemAlloc(&dev, count * sizeof(T)) == 0;
    }
    bool upload(const T* host) { return s->cuMemcpyHtoD(dev, host, n * sizeof(T)) == 0; }
    bool download(T* host)     { return s->cuMemcpyDtoH(host, dev, n * sizeof(T)) == 0; }
};

// ---- the (opt-in, UNVERIFIED) device radix kernel -------------------------
// One LSD pass: histogram with atomics, then a host prefix-sum, then an atomic
// scatter into the output buffer.  Repeats for every 8-bit digit.  This is the
// structure DESIGN.md section 2.6 describes; it is NOT run in CI (no GPU) and
// is provided so a GPU owner can enable FYX_GPU_COMPUTE and debug it there.
#if defined(FYX_GPU_COMPUTE)
inline std::string gpu_radix_kernel_src(std::size_t key_bytes) {
    const char* ktype = key_bytes == 8 ? "unsigned long long"
                      : key_bytes == 4 ? "unsigned int"
                      : key_bytes == 2 ? "unsigned short"
                      :                  "unsigned char";
    return std::string(R"CUDA(
extern "C" __global__ void fyx_hist(const )") + ktype + R"CUDA( *__restrict__ in,
                                  unsigned int* __restrict__ hist,
                                  unsigned int shift, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int d = (unsigned int)((in[i] >> shift) & 0xFFu);
    atomicAdd(&hist[d], 1u);
}
extern "C" __global__ void fyx_scatter(const )") + ktype + R"CUDA( *__restrict__ in,
                                    )" + ktype + R"CUDA( *__restrict__ out,
                                    unsigned int* __restrict__ base,
                                    unsigned int shift, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int d = (unsigned int)((in[i] >> shift) & 0xFFu);
    unsigned int pos = atomicAdd(&base[d], 1u);
    out[pos] = in[i];
}
)CUDA";
}
#endif

// ---- dispatch entry (returns true on success, false to fall back to CPU) ---
template <class T, class Comp>
inline bool gpu_sort_dispatch(T* p, std::size_t n, Comp, const Options&) {
    // Only numeric, default-ascending keys can take the GPU radix path.
    if (!radix_supported_v<T> || !is_ascending_v<Comp, T>) return false;
    CudaSyms& s = cuda_syms();
    if (!s.ok) return false;   // no driver -> CPU fallback

#if defined(FYX_GPU_COMPUTE)
    // UNVERIFIED ON THIS BOX (no GPU).  Wrapped so any failure falls back.
    try {
        CUdevice dev = 0;
        CUcontext ctx = nullptr;
        if (s.cuInit(0) != 0) return false;
        if (s.cuDeviceGet(&dev, 0) != 0) return false;
        if (s.cuCtxCreate(&ctx, 0, dev) != 0) return false;

        GpuBuffer<T> d_in, d_out;
        GpuBuffer<unsigned int> d_hist, d_base;
        if (!d_in.alloc(s, n)  || !d_out.alloc(s, n) ||
            !d_hist.alloc(s, 256) || !d_base.alloc(s, 256)) { s.cuCtxDestroy(ctx); return false; }
        if (!d_in.upload(p)) { s.cuCtxDestroy(ctx); return false; }

        const std::size_t digits = sizeof(T);  // 8-bit digits per byte
        const std::size_t threads = 256, blocks = (n + threads - 1) / threads;
        unsigned int nn = static_cast<unsigned int>(n);
        CUmodule mod = nullptr;
        CUfunction fhist = nullptr, fscat = nullptr;
        std::string ptx;
        {
            nvrtcProgram prog = nullptr;
            std::string src = gpu_radix_kernel_src(sizeof(T));
            if (s.nvrtcCreateProgram(&prog, src.c_str(), "fyx_radix", 0, nullptr, nullptr) != 0)
                { s.cuCtxDestroy(ctx); return false; }
            const char* opts[] = { "--gpu-architecture=compute_70" };
            if (s.nvrtcCompileProgram(prog, 1, opts) != 0)
                { s.nvrtcDestroyProgram(&prog); s.cuCtxDestroy(ctx); return false; }
            std::size_t sz = 0; char* buf = nullptr;
            s.nvrtcGetPTX(prog, buf); /* buf points into prog; load below */
            ptx = std::string(buf ? buf : "");
            s.nvrtcDestroyProgram(&prog);
        }
        if (s.cuModuleLoadData(&mod, ptx.c_str()) != 0) { s.cuCtxDestroy(ctx); return false; }
        s.cuModuleGetFunction(&fhist, mod, "fyx_hist");
        s.cuModuleGetFunction(&fscat, mod, "fyx_scatter");

        std::vector<unsigned int> host_hist(256), host_base(256);
        std::vector<T> dbl_buf(n);  // host scratch for the ping-pong
        const T* cur_in = p;        // we copy through dbl_buf on host each pass
        // (device ping-pong uses d_in/d_out; simplified to a single in/out swap)
        for (std::size_t d = 0; d < digits; ++d) {
            unsigned int shift = static_cast<unsigned int>(d * 8);
            std::memset(host_hist.data(), 0, 256 * sizeof(unsigned int));
            if (s.cuMemcpyHtoD(d_hist.dev, host_hist.data(), 256 * sizeof(unsigned int)) != 0) break;
            void* hargs[] = { &d_in.dev, &d_hist.dev, &shift, &nn };
            s.cuLaunchKernel(fhist, blocks,1,1, threads,1,1, 0, nullptr, hargs, nullptr);
            if (s.cuMemcpyDtoH(host_hist.data(), d_hist.dev, 256 * sizeof(unsigned int)) != 0) break;
            unsigned int sum = 0;
            for (int b = 0; b < 256; ++b) { host_base[b] = sum; sum += host_hist[b]; }
            if (s.cuMemcpyHtoD(d_base.dev, host_base.data(), 256 * sizeof(unsigned int)) != 0) break;
            void* sargs[] = { &d_in.dev, &d_out.dev, &d_base.dev, &shift, &nn };
            s.cuLaunchKernel(fscat, blocks,1,1, threads,1,1, 0, nullptr, sargs, nullptr);
            // swap in/out for next digit
            CUdeviceptr tmp = d_in.dev; d_in.dev = d_out.dev; d_out.dev = tmp;
        }
        if (s.cuMemcpyDtoH(const_cast<T*>(cur_in), d_in.dev, n * sizeof(T)) != 0) { s.cuCtxDestroy(ctx); return false; }
        (void)dbl_buf;
        s.cuCtxDestroy(ctx);
        return true;   // GPU path completed
    } catch (...) {
        return false;  // any failure -> CPU fallback
    }
#else
    (void)p; (void)n;
    return false;      // compute path disabled: CPU fallback (the documented default)
#endif
}

} // namespace detail
} // namespace fyx

#endif // FYX_ENABLE_GPU
