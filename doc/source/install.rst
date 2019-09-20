Installation notes
==================

The `cg_openmm <https://github.com/shirtsgroup/cg_openmm>`_ package will eventually be available for installation via `Anaconda <https://www.anaconda.com/>`_.  This will make resolution of software conflicts much easier.  However, at present, because the package has not been made public, Anaconda installation is not yet possible, and software conflicts must be resolved by the individual.

Here we provide installation instructions which have been tested on multiple platforms.

Dependencies for the cg_openmm package
--------------------------------------

The 'cg_openmm' package was written in order to interface with a separate software package called `foldamers <https://github.com/shirtsgroup/foldamers>`_.  Due to conflicts among dependencies for the foldamers package with other Python versions, **we recommend using Python version 3.6 when installing/using cg_openmm**.

The following is a list of software dependencies for the 'cg_openmm' package, with recommended version numbers in parentheses:

1) `foldamers <https://github.com/shirtsgroup/foldamers>`_ (version 0.0)

    **Dependencies for the 'foldamers' software package:**
        2) `pymbar <https://github.com/choderalab/pymbar>`_ (version 3.0.3)
        3) `MDTraj <http://mdtraj.org/1.9.3/>`_ (version 1.9.3)
        4) `MSMBuilder <http://msmbuilder.org/3.8.0/>`_ (version 3.8)
        5) `scikit-learn <https://scikit-learn.org/stable/>`_ (version 0.18.1)
        6) `sklearn-genetic <https://github.com/manuel-calzolari/sklearn-genetic>`_ (version 0.2)
        7) `kHelios <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00721>`_

8) `OpenMM <http://openmm.org/>`_ (version 7.3.1)
9) `Yank <http://getyank.org/latest/>`_ (version 0.24.1)

Recommended installation steps
------------------------------

We recommend installation of `Anaconda <https://www.anaconda.com/>`_ prior to installation of the 'cg_openmm' package, as this makes resolution of conflicts between dependencies much easier.

We direct users that have not installed `Anaconda <https://www.anaconda.com/>`_ to the `Download page <https://www.anaconda.com/distribution/>`_ in order to select the appropriate version for your platform (Windows, Linux, Mac).  (It shouldn't matter which version of Anaconda is installed.)

The following installation steps are recommended for users that have already installed `Anaconda <https://www.anaconda.com/>`_ on their system:

1) Install 'sklearn-genetic' (used for genetic algorithm optimization of model parameters)::

	pip install sklearn-genetic

2) Create an Anaconda environment for Python version 3.6 (the most stable Python version for 'cg_openmm')::

	conda create -n cg_openmm python=3.6 mdtraj=1.9.3 openmm=7.3.1 pymbar=3.0.3 msmbuilder=3.8 scikit-learn=0.18.1 yank=0.21.2

3) Activate this new environment::

	conda activate cg_openmm

4) Clone and install a fresh copy of 'foldamers'::

	git clone https://github.com/shirtsgroup/foldamers.git
	cd foldamers
	python setup.py install

5) Clone and install a fresh copy of 'cg_openmm' to your system::

	git clone https://github.com/shirtsgroup/cg_openmm.git
	cd cg_openmm
        python setup.py install

