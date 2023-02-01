# pylint: disable=missing-module-docstring
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hrosailing",
    version="0.10.0",
    author="Valentin Dannenberg & Robert Schueler",
    author_email="valentin.dannenberg2@uni-rostock.de",
    description="Python library for Polar (Performance) Diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hrosailing/hrosailing",
    project_urls={
        "Bug Tracker": "https://github.com/hrosailing/hrosailing/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=[
        "hrosailing",
        "hrosailing.cruising",
        "hrosailing.polardiagram",
        "hrosailing.pipeline",
        "hrosailing.pipelinecomponents",
        "hrosailing.pipelinecomponents.modelfunctions",
    ],
    python_requires=">=3.7",
)
