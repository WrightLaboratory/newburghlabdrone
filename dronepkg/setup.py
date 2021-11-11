from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dronepkg",
    version="1.0",
    author="Willy Tyndall",
    author_email="will.tyndall@yale.edu",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"dronepkg": "dronepkg"},
    packages=setuptools.find_packages(include=["dronepkg", 'dronepkg.*']),
    python_requires=">=3.6",
)