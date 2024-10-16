from torch.utils.cpp_extension import load


class KernelLibray():
    def __init__(self):
        self.built = False
        self.library = None
        self.build_kernels()

    def get_library(self):
        return self.library

    def build_kernels(self):
        if (self.built is False):
            self.library = load(name="kernelimpl", sources=["cpp_interfaces.cpp"],
                extra_cflags=["-DPPLNN_CUDA_ENABLE_CUDNN"],
                extra_cuda_cflags=["-U__CUDA_NO_HALF_OPERATORS__",
                                   "-U__CUDA_NO_HALF_CONVERSIONS__",
                                   "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                   "-U__CUDA_NO_HALF2_OPERATORS__"],
                extra_ldflags=["/mnt/llm/workspace/jianliheng/ppl.llm.kernel.cuda/ppl-build/ppl.kernel.cuda-build/libpplkernelcuda_static.a",
                               "/mnt/llm/workspace/jianliheng/ppl.llm.kernel.cuda/ppl-build/ppl.kernel.cuda-build/pplcommon-build/libpplcommon_static.a",
                               "/mnt/hpc/share/jianliheng/cudnn-linux-x86_64-8.9.6.50_cuda11-archive/lib/libcudnn.so"],
                extra_include_paths=["/mnt/llm/workspace/jianliheng/ppl.llm.kernel.cuda/include/",
                                     "/mnt/llm/workspace/jianliheng/ppl.llm.kernel.cuda/deps/pplcommon/src/",
                                     "/mnt/hpc/share/jianliheng/cudnn-linux-x86_64-8.9.6.50_cuda11-archive/include/"],
                verbose=True, with_cuda=True)

            self.built = True

kernel_library = KernelLibray()
module = kernel_library.get_library()
