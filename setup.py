from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

cuda_home = '/usr/local/cuda'

setup(
    name='my_cuda_extension',
    ext_modules=[
        CppExtension('my_cuda_extension', ['gating.cpp'],
                    include_dirs=[f'{cuda_home}/include'],
                    library_dirs=[f'{cuda_home}/lib64' ],
                    extra_link_args=['-L/mnt/llm/workspace/xulei/ws/gating_cpp', '-lcuda_static_lib'], 
                    libraries=['cudart'],
                    #extra_compile_args={'cxx': [], 'nvcc': []},  # 根据需要添加编译参数
                    
                    )
    ],
    cmdclass={'build_ext': BuildExtension}
)
