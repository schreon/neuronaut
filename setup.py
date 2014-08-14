# -*- coding: utf-8-*-
from distutils.core import setup

setup(
    name='neuronaut',
    version='0.0.0',
    author='Leon Schröder',
    author_email='schreon.loeder@gmail.com',
    packages=['neuro', 'neuro.cuda'],
    url='https://github.com/schreon/neuronaut',
    license='LICENSE',
    description='neural network library accelerated by CUDA and OpenCL.',
    long_description=open('README.md').read(),
    install_requires=[]
)