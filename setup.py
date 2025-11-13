"""Setup script for AI Assistant Pro"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-assistant-pro",
    version="0.1.0",
    author="AI Assistant Pro Team",
    description="High-performance AI assistant framework optimized for NVIDIA Blackwell (SM120)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-assistant-pro/ai-assistant-pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=24.0.0",
            "ruff>=0.3.0",
            "mypy>=1.8.0",
        ],
        "benchmarks": [
            "matplotlib>=3.8.0",
            "pandas>=2.2.0",
            "seaborn>=0.13.0",
        ],
    },
)
