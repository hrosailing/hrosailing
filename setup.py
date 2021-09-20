import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hrosailing",
    version="0.0.1",
    author="Valentin Dannenberg / Robert Schueler",
    author_email="valentin.dannenberg2@uni-rostock.de",
    description="Python library for Polar (Performance) Diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VFDannenberg/hrosailing",
    project_urls={
        "Bug Tracker": "https://github.com/VFDannenberg/hrosailing/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: PyPy"
        "License :: ",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "hrosailing"},
    packages=setuptools.find_packages(where="hrosailing"),
    python_requires=">=3.8",
)
