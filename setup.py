# -*- coding: iutf-8-*-
from distutils.core import setup

setup(
    name='neuronaut',
    version='0.0.0',
    author='Leon Schröder',
    author_email='schreon.loeder@gmail.com',
    packages=['neuro'],
    url='https://github.com/schreon/neuronaut',
    license='LICENSE',
    description='Useful towel-related stuff.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy == 1.8.1",
        "reikna >= 0.6.3",
    ],
)