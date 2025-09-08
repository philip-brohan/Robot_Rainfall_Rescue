How to reproduce and extend this work
=====================================

This project is designed to be easy to reproduce and extend. Everything involved is kept under version control in a `git repository <https://en.wikipedia.org/wiki/Git>`_. The repository is hosted on `GitHub <https://github.com/>`_ (and the documentation made with `GitHub Pages <https://pages.github.com/>`_). The repository is `<https://github.com/philip-brohan/Robot_Rainfall_Rescue>`_. This repository contains everything you need to reproduce or extend this work.

If you are familiar with GitHub, you already know what to do (fork or clone `the repository <https://github.com/philip-brohan/Robot_Rainfall_Rescue>`_): If you'd prefer not to bother with that, you can download the whole thing as `a zip file <https://github.com/philip-brohan/Robot_Rainfall_Rescue/archive/master.zip>`_.

As well as downloading the software, some setup is necessary to run it successfully:

These scripts will only work in a `python <https://www.python.org/>`_ environment with the appropriate python version and libraries available. I use `conda <https://docs.conda.io/en/latest/>`_ to manage the required python environment - which is specified in a yaml file:

.. literalinclude:: ../environments/rrr-spice.yml


As always with ML work, you will need access to a GPU. I use a dedicated `MS Azure ML workspace <https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace?view=azureml-api-2>`_ for this, so the code in this repository contains some scripts and configuration files specific to that. But the actual training and analysis code is not specific to Azure, and should work in any suitable python environment with access to a GPU (I used 80Gb H100s, mostly; 60Gb A100s will do instead. I have not tested any of this on smaller GPUs). All the scripts here will run on a single 60Gb A100.


The project documentation (these web pages) are included in the repository (in the `docs directory <https://github.com/philip-brohan/Robot_Rainfall_rescue/tree/master/docs>`_). The documentation is in `reStructuredText <https://en.wikipedia.org/wiki/ReStructuredText>`_ format, and uses the `Sphinx <https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html>`_ documentation generator.


