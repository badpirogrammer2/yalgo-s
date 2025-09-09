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
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.0.0",
        "numpy",
        "pillow",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/badpirogrammer2/yalgo-s/issues",
        "Source": "https://github.com/badpirogrammer2/yalgo-s",
    },
)
