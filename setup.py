import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causalty",
    version="0.0.1",
    author="Byron Galbraith",
    author_email="byron.galbraith@gmail.com",
    description="Causality Analysis Tools for Neuroimaging Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bgalbraith/causalty",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires='>=3.6',
)
