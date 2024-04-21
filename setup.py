import setuptools
from setuptools import find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _read_reqs(relpath):
    fullpath = path.join(path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]


REQUIREMENTS = _read_reqs("requirements.txt")

setuptools.setup(
    name="open_lm",
    version="0.0.34",
    author=[
        "Suchin Gururangan*",
        "Mitchell Wortsman*",
        "Samir Yitzhak Gadre",
        "Achal Dave",
        "Maciej Kilian",
        "Weijia Shi",
        "Georgios Smyrnis",
        "Gabriel Ilharco",
        "Matt Jordan",
        "Ali Farhadi",
        "Ludwig Schmidt",
    ],
    author_email="sg01@cs.washington.edu",
    description="OpenLM",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=REQUIREMENTS,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlfoundations/open_lm",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
)
