from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beamcals",
    version="1.0",
    author="Will Tyndall",
    author_email="will.tyndall@yale.edu",
    description="BEAm Mapping CALibration System",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"beamcals": "beamcals"},
    packages=setuptools.find_packages(include=["beamcals", 'beamcals.*']),
    python_requires=">=3.7",
)