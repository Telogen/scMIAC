# setup.py 示例
from setuptools import setup, find_packages

setup(
    name='scMIAC',
    version='1.0.0',
    author='Lejin Tian',
    author_email='ljtian20@fudan.edu.cn',
    description='scMIAC: Single-Cell Multi-modality Integration via cell type filtered Anchors using Contrastive learning',
    url="https://github.com/Telogen/scMIAC", 
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "anndata",
        "joblib>=1.4.2",
        "matplotlib>=3.7.5",
        "numba>=0.58.1",
        "numpy>=1.24.4",
        "pandas>=2.0",
        "pynndescent>=0.5.13",
        "scanpy",
        "scib>=1.1.5",
        "scikit_learn>=1.3.2",
        "scipy",
        "seaborn>=0.13.2",
        # "setuptools==75.3.0",
        # "torch==1.10.1+cu111",
        "typing_extensions==4.12.2",
    ],
    python_requires='>=3.8',
)
