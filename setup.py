from setuptools import find_packages, setup

setup(
    name="pooling",
    version="1.0.0",
    author="anonymous",
    author_email="anonymous",
    description="Code repository for reproducing results from the `Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs` paper",
    long_description="Code repository for reproducing results from the `Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs` paper",
    long_description_content_type="text/markdown",
    url="",
    packages=[package for package in find_packages() if package.startswith("pooling")],
    zip_safe=False,
    install_requires=[
        "torch",
        "numpy",
        "pettingzoo[mpe]",
        "opencv-python",
        "dm_tree",
        "typer",
        "scipy",
        "lz4",
        "tensorboard",
        "pyvirtualdisplay",
        "gymnasium==0.29.0",
        "ray==2.37.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.11",
)
