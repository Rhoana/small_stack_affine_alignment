import setuptools

setuptools.setup(
    description="A small stack affine alignment process",
    install_requires=[
        "pyaml>=15.8.2",
        "numpy>=1.12.0",
        "scipy>=0.15.0",
        "opencv-python>=3.1.0"
        ],
    name="small_stack_affine_alignment",
    packages=["small_stack_affine_alignment"],
    url="https://github.com/Rhoana/small_stack_affine_alignment",
    version="0.1.0"
)
