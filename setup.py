#!/usr/bin/env python
"""Setup file for the Python datarray package."""

import os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

# Commit to setuptools
import setuptools

from distutils.core import setup

# Get version and release info, which is all stored in datarray/version.py
ver_file = os.path.join('datarray', 'version.py')
# Use exec on contents for Python 3 compatibility
with open(ver_file, 'rt') as fobj:
    exec(fobj.read())

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            requires=REQUIRES,
            install_requires=INSTALL_REQUIRES,
            zip_safe = False,
            )


# Now call the actual setup function
if __name__ == '__main__':
    setup(**opts)
