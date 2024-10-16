import re
from setuptools import setup, find_packages

def get_version():
    """Get package version from info.py file"""
    filename = "abstractdecoder/info.py"
    with open(filename, encoding="utf-8") as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError(f"{filename} doesn't contain __version__")
    version = match.groups()[0]
    return version

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="abstractdecoder",
    version=get_version(),
    author="Duong Nguyen",
    author_email="duongng2911@gmail.com",
    description="An NLP-powered tool designed to classify each sentence of a clinical trial abstract into its specific role.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duongnguyen-dev/abstract-to-skim/tree/main",
    license='Apache License 2.0',
    packages=find_packages(),
    classifiers=[
        "Operating System :: OS Independent",
        'License :: OSI Approved :: Apache Software License',
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn==1.5.2",
        "tensorflow==2.17.0",
        "tensorflow-text==2.17.0",
        "tensorflow-hub==0.16.1",
        "loguru==0.7.2",
        "pandas==2.2.3"
    ],
)