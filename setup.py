from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="universal-data-pipeline",
    version="1.0.0",
    author="Universal Pipeline Team",
    author_email="contact@universalpipeline.com",
    description="A comprehensive data processing pipeline for multiple data types",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/universal-data-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "all": [
            "ultralytics",  # For YOLO integration
            "xgboost",      # For advanced ML models
            "lightgbm",     # For gradient boosting
            "catboost",     # For categorical boosting
            "prophet",      # For time series forecasting
            "statsmodels",  # For ARIMA models
        ],
    },
    entry_points={
        "console_scripts": [
            "universal-pipeline=scripts.cli_processor:main",
            "up-process=scripts.quick_process:main",
            "up-analyze=scripts.analyze_dataset:main",
        ],
    },
    include_package_data=True,
    package_data={
        "universal_pipeline": ["*.json", "*.yaml"],
        "configs": ["*.json", "*.yaml"],
    },
) 