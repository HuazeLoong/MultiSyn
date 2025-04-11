from setuptools import setup, find_packages

setup(
    name="multisyn",
    version="0.1.0",
    description="Multisyn: A multi-source fusion framework for synergistic drug combination prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Huaze Loong",
    author_email="loong@wust.edu.cn",
    url="https://github.com/HuazeLoong/MultiSyn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    keywords=["drug synergy", "GNN", "multimodal", "bioinformatics", "drug discovery"],
    classifiers=[
        "Development Status :: 4 - Beta",  # 可选：1-Planning, 2-Pre-Alpha, 3-Alpha, 4-Beta, 5-Production/Stable
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.0.0",
        "dgl>=1.1.0",
        "rdkit>=2022.9.5",
        "scikit-learn>=1.2.0",
        "numpy>=1.24.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm",
        "matplotlib",
    ],
)
