#! /usr/bin/env python
import os

from setuptools import setup


descr = """a EEG dataset I/O that follows the BIDS data structure standard as much as possible."""

DISTNAME = 'nice_bids'
DESCRIPTION = descr
MAINTAINER = '@laouen'
MAINTAINER_EMAIL = 'laouen.belloli@gmail.com'
URL = 'https://github.com/laouen/nice-bids'
LICENSE = 'Copyright'
DOWNLOAD_URL = 'https://github.com/laouen/nice-bids'
VERSION = '1.0.0'


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        long_description=open('README.md').read(),
        zip_safe=True,  # the package can run out of an .egg file
        classifiers=['Intended Audience :: Science/Research',
                    'Intended Audience :: Developers',
                    'License :: OSI Approved',
                    'Programming Language :: Python',
                    'Topic :: Software Development',
                    'Topic :: Scientific/Engineering',
                    'Operating System :: Microsoft :: Windows',
                    'Operating System :: POSIX',
                    'Operating System :: Unix',
                    'Operating System :: MacOS'],
        platforms='any',
        packages=setuptools.find_packages(exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests"
        ]),
        install_requires=[
            'pandas'
        ],
        package_data={},
        scripts=[]
    )
