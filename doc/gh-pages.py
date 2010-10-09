#!/usr/bin/env python
"""Script to commit the doc build outputs into the github-pages repo.

Use:

  gh-pages.py [tag]

If no tag is given, the current output of 'git describe' is used.  If given,
that is how the resulting directory will be named.

In practice, you should use either actual clean tags from a current build or
something like 'current' as a stable URL for the most current version of the """

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import os
import re
import shutil
import sys
from os import chdir as cd
from os.path import join as pjoin

from subprocess import Popen, PIPE, CalledProcessError, check_call

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

gh_pages = '../../datarray-doc'
html_dir = 'build/html'

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)


def sh2(cmd):
    """Execute command in a subshell, return stdout.

    Stderr is unbuffered from the subshell.x"""
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip()


def sh3(cmd):
    """Execute command in a subshell, return stdout, stderr

    If anything appears in stderr, print it out to sys.stderr"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = p.communicate()
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip(), err.rstrip()


def render_htmlindex(fname, tag):
    rel = '<li> Release: <a href="{t}/index.html">{t}</a>'.format(t=tag)
    rep = re.compile('<!-- RELEASE -->')
    out = []
    with file(fname) as f:
        for line in f:
            out.append(line)
            if rep.search(line):
                out.append(rep.sub(rel, line))
    return ''.join(out)


def new_htmlindex(fname, tag):
    new_page = render_htmlindex(fname, tag)
    os.rename(fname, fname+'~')
    with file(fname, 'w') as f:
        f.write(new_page)


#-----------------------------------------------------------------------------
# Script starts
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # The tag can be given as a positional argument
    try:
        tag = sys.argv[1]
    except IndexError:
        tag = sh2('git describe')
        
    startdir = os.getcwd()
    dest = pjoin(gh_pages, tag)

    sh('make html')
    
    # This is pretty unforgiving: we unconditionally nuke the destination
    # directory, and then copy the html tree in there
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(html_dir, dest)

    try:
        cd(gh_pages)
        status = sh2('git status | head -1')
        branch = re.match('\# On branch (.*)$', status).group(1)
        if branch != 'gh-pages':
            e = 'On %r, git branch is %r, MUST be "gh-pages"' % (gh_pages,
                                                                 branch)
            raise RuntimeError(e)

        sh('git add %s' % tag)
        new_htmlindex('index.html', tag)
        sh('git add index.html')
        sh('git commit -m"Created new doc release, named: %s"' % tag)
        print
        print 'Most recent 3 commits:'
        sys.stdout.flush()
        sh('git --no-pager log --oneline HEAD~3..')
    finally:
        cd(startdir)

    print
    print 'Now verify the build in: %r' % dest
    print "If everything looks good, 'git push'"
