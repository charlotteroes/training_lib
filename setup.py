from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

cuda_home = '/usr/local/cuda'

setup(
    name='my_cuda_extension',
    ext_modules=[
        CppExtension('my_cuda_extension', ['ppl_deepseek_kernel.cpp'],
                    include_dirs=[f'{cuda_home}/include'],
                    library_dirs=[f'{cuda_home}/lib64' ],
                    extra_link_args=['-L.', 
                                    #  '-L/mnt/llm/workspace/xupinjie/software/miniconda3/lib/python3.12/site-packages/torch/include', 
                                    #  '-L/mnt/llm/workspace/xupinjie/software/miniconda3/lib/python3.12/site-packages/torch/lib',
                                     '-lcuda_static_lib', 
                                    #  '-ltorch',
                                    #  '-ltorch_cuda',
                                     ], 
                    libraries=['cudart'],
                    #extra_compile_args={'cxx': [], 'nvcc': []},  # 根据需要添加编译参数
                    # extra_compile_args={
                    #     'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=1'],  # 或 1
                    # }
                    ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
