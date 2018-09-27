# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

long_description = open("README.md").read()

setup(
    name="place-stimulation",
    packages=find_packages(),
    version='0.1',
    include_package_data=True,
    author="",
    author_email="",
    maintainer="",
    maintainer_email="",
    platforms=['Linux', "Windows"],
    description="Plugins for the place-stimulation project",
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'plugin-expipe-place-stimulation = place_stimulation.cli.main:reveal'
        ],
    },
)
