from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

cuda_sources = [
    os.path.join(csrc_dir, "bindings.cpp"),
    os.path.join(src_dir,  "reference.cu"),
    os.path.join(src_dir,  "fused_decode.cu"),
    os.path.join(src_dir,  "multi_head.cu"),
]

nvcc_flags = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-std=c++17",
]

setup(
    name="linear_decode",
    version="0.1.0",
    description="Fused recurrent linear attention decode: smem-resident state with gated rank-1 updates",
    ext_modules=[
        CUDAExtension(
            name="linear_decode._C",
            sources=cuda_sources,
            include_dirs=[src_dir, csrc_dir],
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["linear_decode"],
    python_requires=">=3.8",
)
