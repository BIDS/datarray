"""
:Description: Sphinx extension to remove leading under-scores from directories names in the html build output directory.
"""

############################################################################
# NOTE
#
# Other than this comment, this file was copied verbatim from the
# github-tools project by Damien Lebrun:
#
# http://github.com/dinoboff/github-tools
#
# We renamed it from sphinx.py to sphinx_no_underscore.py but have made no
# changes to the code.
#
# The original file is licensed under the terms of the BSD license, as
# indicated here:
# http://github.com/dinoboff/github-tools/blob/master/LICENCE
#
# This copy was made at revision:
# http://github.com/dinoboff/github-tools/blob/dba0afe75d2f3388d4b3c21d52f2bf0f2a312e1c/src/github/tools/sphinx.py
#
# End of NOTE
############################################################################

import os
import shutil


def setup(app):
    """
    Add a html-page-context  and a build-finished event handlers
    """
    app.connect('html-page-context', change_pathto)
    app.connect('build-finished', move_private_folders)
                
def change_pathto(app, pagename, templatename, context, doctree):
    """
    Replace pathto helper to change paths to folders with a leading underscore.
    """
    pathto = context.get('pathto')
    def gh_pathto(otheruri, *args, **kw):
        if otheruri.startswith('_'):
            otheruri = otheruri[1:]
        return pathto(otheruri, *args, **kw)
    context['pathto'] = gh_pathto
    
def move_private_folders(app, e):
    """
    remove leading underscore from folders in in the output folder.
    
    :todo: should only affect html built
    """
    def join(dir):
        return os.path.join(app.builder.outdir, dir)
    
    for item in os.listdir(app.builder.outdir):
        if item.startswith('_') and os.path.isdir(join(item)):
            shutil.move(join(item), join(item[1:]))
