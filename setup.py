# -*- coding: utf-8 -*-

# Learn more: https://github.com/mzahana/octune

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='octune',
    version='0.1.0',
    description='Backprobagation-based online linear control tuning.',
    long_description=readme,
    author='Mohamed Abdelkader',
    author_email='mohamedashraf123@gmail.com',
    url='https://github.com/mzahana/online_control_tuning',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)