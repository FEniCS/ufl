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


Preliminaries
=============

Installation
------------

Ubuntu package
^^^^^^^^^^^^^^

UFL may be installed directly from source, but the Debian (Ubuntu)
package ``python-ufl`` is also available for UFL, as for other FEniCS
components.

Manual from source
^^^^^^^^^^^^^^^^^^

To retrieve the latest development version of UFL::

    git clone https://bitbucket.org/fenics-project/ufl

To install UFL::

    python setup.py install

Help and support
----------------

Send help requests and questions to fenics-support@googlegroups.com.

Send feature requests and questions to fenics-dev@googlegroups.com


Development and reporting bugs
------------------------------

The git source repository for UFL is located at
https://bitbucket.org/fenics-project/ufl. For general UFL development
questions and to make feature requests, use fenics-dev@googlegroups.com

Bugs can be registered at
https://bitbucket.org/fenics-project/ufl/issues.


Manual and API Reference
========================

.. toctree::
   :titlesonly:

   User Manual <user/user_manual>
   API Reference <api-doc/ufl>
   Releases <releases>


* :ref:`modindex`
* :ref:`genindex`
* :ref:`search`
