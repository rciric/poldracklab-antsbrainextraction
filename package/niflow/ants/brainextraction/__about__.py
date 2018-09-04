# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
__version__ = '0.0.1-dev'
__packagename__ = 'niflow-ants-brainextraction'
__author__ = 'The CRN developers'
__copyright__ = 'Copyright 2018, Center for Reproducible Neuroscience, Stanford University'
__credits__ = ['Oscar Esteban', 'Christopher J. Markiewicz', 'Chris Gorgolewski',
               'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Oscar Esteban'
__email__ = 'crn.poldracklab@gmail.com'
__status__ = 'Prototype'

__description__ = "Nipype implementation of the antsBrainExtraction utility."
__longdesc__ = __description__

DOWNLOAD_URL = (
    'https://github.com/niflows/{name}/archive/{version}.tar.gz'.format(
        name=__packagename__, version=__version__))
URL = 'https://github.com/niflows/{}'.format(__packagename__)
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

REQUIRES = [
    'nipype',  # Unknown minimum version
    'packaging',
]

SETUP_REQUIRES = []
REQUIRES += SETUP_REQUIRES

LINKS_REQUIRES = []
TESTS_REQUIRES = []

EXTRA_REQUIRES = {
    'tests': TESTS_REQUIRES,
}


def _list_union(iterable):
    return list(set(sum(iterable, [])))

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = _list_union(EXTRA_REQUIRES.values())
