neuronaut
=========

neuronaut is a neural network library written in Python. It has backends for OpenCL and CUDA.
neuronaut is meant to be fast, easy to use and easy to extend.

Dependencies
============
neuronaut is based on numpy and [reikna ](https://github.com/Manticore/reikna). reikna again needs [pycuda](https://github.com/inducer/pycuda) and/or [pyopencl](https://github.com/inducer/pyopencl) to run.

Installation
============
This repository is still private, so you need to send me your public key. Then you can install neuronaut using pip:
`pip install git+ssh://git@github.com/schreon/neuronaut.git` It is highly recommended to use [virtualenv](https://pypi.python.org/pypi/virtualenv).


Getting started
===============
- choose backend + create context
- upload numpy arrays
- create network
- create trainer
- create evaluator

Available mixins
=====================
stub

Write your own mixins
=====================
stub

Write your own activation function
==================================
stub
