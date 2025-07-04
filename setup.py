from setuptools import setup, find_packages

setup(
    name="multi-horizon-glucose-prediction",
    version="0.1.0",
    author="Najmeh Mohajeri",
    author_email="nmohajeri@gmail.com",
    description="Multi-horizon glucose prediction using LSTM and CNN-LSTM models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NajmehM/multi-horizon-glucose-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "tensorflow>=2.10.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black>=22.0", "flake8>=4.0"],
        "notebook": ["jupyter>=1.0.0", "notebook>=6.4.0"],
    },
)