# setup.py 示例
from setuptools import setup, find_packages

setup(
    name='scMIAC',
    version='1.0.0',
    author='Lejin Tian',
    author_email='ljtian20@fudan.edu.cn',
    description='scMIAC: Single-Cell Multi-modality Integration via cell type filtered Anchors using Contrastive learning',
    packages=find_packages(),
    install_requires=[
        # 依赖的其他包
    ],
    python_requires='>=3.8',
)
