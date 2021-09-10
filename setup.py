import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hrosailing",
    version="0.0.1",
    author="Valentin Dannenberg / Robert Schueler",
    author_email="valentin.dannen@googlemail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VFDannenberg/hrosailing",
    project_urls={
        "Bug Tracker": "https://github.com/VFDannenberg/hrosailing/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: ",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "hrosailing"},
    packages=setuptools.find_packages(where="hrosailing"),
    python_requires=">=3.8",
)
