scikit-protopy
==============

Prototype Selection and Generation Toolbox based on scikit-learn.

This project was started in 2014 by Dayvid Victor as a result of
his masters and on-going PhD at Federal University of Pernambuco.

The aim of this project is to provide Prototype Selection (PS) and 
Prototype Generation (PG) techniques to be applied where instance 
reduction is needed (noisy sensitive domains, imbalanced datasets, 
high density clusters ...). 

Right now, this project is referenced in the scikit-learn official 
wiki (https://github.com/scikit-learn/scikit-learn/wiki/Third-party-projects-and-code-snippets). 
Hopefully, the PS/PG techniques might be included in the scikit-learn 
project anytime soon.


Dependencies
============

This is designed to work with:
- Python 2.6+
- scikit-learn == 0.16.0
- Numpy >= 1.3
- SciPy >= 0.7
- Matplotlib >= 0.99.1. (for examples, only)


Install
=======

To install, use::

    sudo python setup.py install

To install via easy\_install, use::

    sudo easy_install .


Important References
====================

For all algorithms a reference is provided. But if you are new to
PS/PG, we recommend the following papers:

- **Prototype Selection**: Garcia, Salvador, et al. "Prototype selection for nearest neighbor classification: Taxonomy and empirical study." Pattern Analysis and Machine Intelligence, IEEE Transactions on 34.3 (2012): 417-435.

- **Prototype Generation**: Triguero, Isaac, et al. "A taxonomy and experimental study on prototype generation for nearest neighbor classification." Systems, Man, and Cybernetics, Part C: Applications and Reviews, IEEE Transactions on 42.1 (2012): 86-100.





