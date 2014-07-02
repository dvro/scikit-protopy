"""
Prototype Selection and Generation module for Python
====================================================

protopy is a Python module integrating classical instance reduction
algorithms in the tightly-knit world of scientific Python packages 
(numpy, scipy, matplotlib, sklearn).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See: https://github.com/dvro/scikit-protopy for complete documentation.

The two main modules are: 
 - selection: techniques that generate a smaller training set, selecting
   a subset of instances based a specific rule/heuristic.
 - generation: techniques that generate a smaller training set, generating
   new instances that do not necessarilly belongs to the original 
   training set.


"""


__all__ = ['selection', 'generation']
