from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="abstract-to-skim",
    version="0.1.0",
    author="Duong Nguyen",
    author_email="duongng2911@gmail.com",
    description="Turning paper's abstract into the format that more skimmable.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duongnguyen-dev/abstract-to-skim/tree/main",
    packages=find_packages(),
    classifiers=[],
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn==1.5.2",
        "tensorflow==2.17.0",
        "tensorflow-text==2.17.0",
        "tensorflow-hub==0.16.1",
        "loguru==0.7.2",
        "pandas==2.2.3"
    ],
    entry_points={
        "console_scripts": [
            "abstract-to-skim train=abstract_to_skim.train:main",
        ]
    }
)