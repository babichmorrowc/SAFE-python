"""A Python implementation of SAFE"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

from io import open

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='safepython',  # Required
    version='0.2.0',  # Required
    description='A Python implementation of the SAFE toolbox for sensitivity analysis',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://safetoolbox.github.io',  # Optional
    author='Francesca Pianosi, Fanny Sarrazin, Thorsten Wagener',  # Optional
    author_email='fanny.sarrazin@inrae.fr',  # Optional
    license='GPL-3.0', # Optional
    packages=find_packages(where="src",exclude=['examples']),  # Required
    package_dir={"": "src"},
#    python_requires='>=3.7, <4',
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.3.0",
        "matplotlib>=2.2.3",
    ],
)
