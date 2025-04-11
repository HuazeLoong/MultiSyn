from setuptools import setup, find_packages

setup(
    name="multisyn",
    version="0.1.0",
    description="A multi-source information fusion framework for synergistic drug combination prediction.",
    url="https://github.com/HuazeLoong/MultiSyn",
    packages=find_packages(where="src"),  # <-- Use the src directory as the package source
    package_dir={"": "src"},              # <-- Explicitly specify src as the root path of the package
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.0.0",
        "dgl>=1.1.0",
        "rdkit>=2022.9.5",
        "scikit-learn>=1.2.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ]
)
