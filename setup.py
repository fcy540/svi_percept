from setuptools import setup, find_packages

setup(
    name="svi_percept",
    version="0.1.0",
    description="Model of street view imagery perception using CLIP",
    author="Matthew Danish",
    author_email="matthewrdanish@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "open_clip_torch>=2.20.0",
        "pillow>=8.0.0",
        "numpy>=1.19.0"
    ],
)

