# -*- coding: utf-8 -*-

# Learn more: https://github.com/mzahana/octune
# Reference for Python package structuring (https://www.pythonforthelab.com/blog/how-create-setup-file-your-project/)

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='octune',
    version='0.1.0',
    description='Backprobagation-based  data-driven online linear control tuning for unknown system.',
    long_description=readme,
    author='Mohamed Abdelkader',
    author_email='mohamedashraf123@gmail.com',
    url='https://github.com/mzahana/octune',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points={
        'console_scripts': [
            'octune_main=octune.__main__:main',
        ]
    },
    install_requires=[
        'numpy==1.21.0'
        "control==0.8.4;python_version<'3.4'",
    ]
)