from setuptools import setup, find_packages

setup(
    name="waveattn_sopseg",
    version="1.0.0",
    description="WaveAttn SOPSeg: High-Precision Mangrove Canopy Disturbance Mapping via Frequency-Aware Prompted Segmentation",
    author="Can Gio Mangrove Disturbance Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "segmentation-models-pytorch>=0.3.3",
        "rasterio>=1.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ]
)
