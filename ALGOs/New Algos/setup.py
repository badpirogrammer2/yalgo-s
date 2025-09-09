from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yalgo-s",
    version="0.1.0",
    author="YALGO-S Team",
    author_email="team@yalgo-s.com",
    description="Yet Another Library for Gradient Optimization and Specialized algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/badpirogrammer2/yalgo-s",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core ML/DL Libraries
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.0.0",

        # Scientific Computing
        "numpy>=1.21.0",
        "scipy>=1.7.0",

        # Data Processing
        "pillow>=8.0.0",
        "pandas>=1.3.0",

        # Machine Learning
        "scikit-learn>=1.0.0",

        # Parallel Processing
        "concurrent.futures; python_version >= '3.2'",

        # Utilities
        "pathlib>=1.0.1",
        "typing>=3.7.4",
    ],
    extras_require={
        "dev": [
            # Testing Framework
            "pytest>=6.0",
            "pytest-cov",
            "hypothesis>=6.0",

            # Code Quality
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "pre-commit>=2.0",

            # Development Tools
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
        "docs": [
            # Documentation
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
            "myst-parser>=0.15",
        ],
        "gpu": [
            # GPU-specific optimizations
            "torch[cuda]>=2.0.0",
            "torch[mps]>=2.0.0; sys_platform == 'darwin'",
            "GPUtil>=1.4",
            "psutil>=5.8",
        ],
        "datasets": [
            # Dataset loading and management
            "datasets>=2.0.0",
            "huggingface-hub>=0.10",
        ],
        "visualization": [
            # Data visualization
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0",
        ],
        "ml": [
            # Additional ML libraries
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0",
            "optuna>=2.10",
        ],
        "deployment": [
            # Web and deployment
            "fastapi>=0.80",
            "uvicorn>=0.18",
            "gunicorn>=20.1",
            "docker>=5.0",
            "kubernetes>=20.0",
        ],
        "cloud": [
            # Cloud SDKs
            "boto3>=1.24",
            "google-cloud-storage>=2.0",
            "azure-storage-blob>=12.0",
            "azure-identity>=1.10",
        ],
        "performance": [
            # Performance optimization
            "numba>=0.56",
            "cython>=0.29",
            "ray>=2.0",
            "dask>=2022.0",
        ],
        "all": [
            # All optional dependencies
            "pytest>=6.0",
            "pytest-cov",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "torch[cuda]>=2.0.0",
            "datasets>=2.0.0",
            "matplotlib>=3.5.0",
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "fastapi>=0.80",
            "uvicorn>=0.18",
            "boto3>=1.24",
            "google-cloud-storage>=2.0",
            "azure-storage-blob>=12.0",
            "numba>=0.56",
            "ray>=2.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/badpirogrammer2/yalgo-s/issues",
        "Source": "https://github.com/badpirogrammer2/yalgo-s",
    },
)
