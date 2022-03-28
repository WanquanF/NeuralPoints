from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamferdis',
    ext_modules=[
        CUDAExtension('chamferdis', [
            'chamfer_distance.cpp',
            'chamfer_distance.cu',
        ],
                extra_compile_args=['-g']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
