.. title:: Unified Form Language

***************************
Unified Form Language (UFL)
***************************

This is the documentation for the Unified Form Language from the
FEniCS Project (http://fenicsproject.org). The Unified Form Language
(UFL) is a domain specific language for declaration of finite element
discretizations of variational forms. More precisely, it defines a
flexible interface for choosing finite element spaces and defining
expressions for weak forms in a notation close to mathematical
notation. UFL is described in the paper

Alnæs, M. S., Logg A., Ølgaard, K. B., Rognes, M. E. and
Wells, G. N. (2014). Unified Form Language: A domain-specific language
for weak formulations of partial differential equations. *ACM
Transactions on Mathematical Software* 40(2), Article 9, 37
pages. [http://dx.doi.org/doi:10.1145/2566630>]
[http://arxiv.org/abs/1211.4047]

UFL is most commonly used as the input language for the FEniCS Form
Compiler (FFC) and in combination with the problem solving environment
DOLFIN.



Installation
============

Debian/Ubuntu packages
----------------------

Debian/Ubuntu
^^^^^^^^^^^^^

A Debian/Ubuntu package ``python-ufl`` is available for UFL:

    sudo apt-get install python-ufl


Ubuntu PPA
^^^^^^^^^^

UFL is available in the FEniCS Project PPA. The version of UFL
available in the PPA will generally more recent than the Debian/Ubuntu
package. To install UFL from the PPA:

    sudo add-apt-repository ppa:fenics-packages/fenics
    sudo apt-get update
    sudo apt-get install fenics


From source
-----------

Dependencies
^^^^^^^^^^^^

UFL depends on the Python packages ``numpy`` and ``six``, and
``setuptools`` is recommended. If ``setuptools`` is available, the UFL
installer will install missing dependencies automatically.


Installation
^^^^^^^^^^^^

The source for UFL releases can be downloaded from
http://fenicsproject.org/pub/software/ufl/. To install UFL
system-wide, from the source directory use:

    python setup.py install

To install into a specified directory, use the ``--prefix`` option.


Help and support
================

Send help requests and questions to fenics-support@googlegroups.com.

Send feature requests and questions to fenics-dev@googlegroups.com


Development and reporting bugs
------------------------------

The git source repository for UFL is located at
https://bitbucket.org/fenics-project/ufl. For general UFL development
questions and to make feature requests, use
fenics-dev@googlegroups.com.

Bugs can be registered at
https://bitbucket.org/fenics-project/ufl/issues.



Manual and API reference
========================

.. toctree::
   :titlesonly:

   User Manual <user/user_manual>
   API Reference <api-doc/modules>
   Releases <releases>

* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
