"""
 Created by Narayan Schuetz
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ax3_pipeline",
    version="0.1.5",
    author="Narayan Schütz",
    author_email="narayan.schuetz@artorg.unibe.ch",
    description="Provides functionality to build 3-axis accelerometer feature extraction pipelines (used with raw Axivity AX3 signals)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=["markdown==3.0.1", "numpy>=1.16.2", "pandas>=0.24.2", "statsmodels>=0.9.0",
                      "scipy==1.2.1", "pyhrv==0.3.2", "matplotlib", "seaborn"],  # if it doesn't work try replacing >= with ==
    test_suite='nose.collector',
    tests_require=['nose', "pandas"],
    zip_safe=False,
    python_requires='>=3'
)