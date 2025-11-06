from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuraxon",
    version="0.1.0",
    author="David Vivancos, Jose Sanchez",
    author_email="vivancos@vivancos.com",
    author_email2="jose.sanchezgarcia@unir.net",
    description="Bio-inspired neural network with trinary states and continuous processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DavidVivancos/Neuraxon",
    py_modules=["neuraxon"],
    python_requires=">=3.7",
    install_requires=[],  # No external dependencies!
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
    ],
)
