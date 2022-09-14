#!/usr/bin/env python
import os

from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as file:
    long_description = file.read()

setup(
    name="sequencelearn",
    version="0.0.8",
    author="Johannes Hötter",
    author_email="johannes.hoetter@kern.ai",
    description="Scikit-Learn like Named Entity Recognition modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/code-kern-ai/sequencelearn",
    keywords=["kern.ai", "machine learning", "supervised learning", "python"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"": "."},
    packages=find_packages("."),
    install_requires=[
        "numpy>=1.22.3",
        "scikit-learn>=1.0.2",
        "scipy>=1.8.0",
        "torch>=1.11.0",
    ],
)
