from setuptools import find_packages, setup

setup(
    name="pooling",
    version="0.0.0",
    author="Greyson Brothers",
    author_email="greysonbrothers@gmail.com",
    description="Code repository for reproducing results from the `Attenuation Networks` paper",
    long_description="Code repository for reproducing results from the `Attenuation Networks` paper",
    long_description_content_type="text/markdown",
    url="",
    packages=[package for package in find_packages() if package.startswith("attenuator")],
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
