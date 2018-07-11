import numpy as np
import os

def isclose(r1,r2,rtol=1e-4):
    return np.linalg.norm(np.subtract(r1,r2)) < rtol


def dict_compare(d1, d2):
    """Compare two dictionaries containing mutable objects.

    This compares to dictionaries. Two dictionaries are considered equal
    even if the position of keys differ. Handles mutable values in the dict.
    Some parts are taken from:

    https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python

    Parameters:

    d1,d2: python dictionaries
        dictionaries to be compared

    Return: boolean
        True if dicts are equal, False if they are different.
    """
    areeq = True

    if len(d1) != len(d2):
        return False

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    if len(d1) != len(intersect_keys):
        return False

    for k in d1_keys:
        for v1,v2 in zip(d1[k],d2[k]):
            if v1 != v2:
                return False

    return areeq


def sub_folders(root):
    """Return list of subfolders

    Parameters:

    root: string
        path of the absolute folder for which you want to get the list of
        subfolders.
    """
    #return os.walk(root).next()[1] #python 2.7
    return next(os.walk(root))[1]

def is_integer(s):
    """Determine whether argument is integer or can be converted to integer

    Parameters:

    s: object
        Can be any object for which we want to determine whether is integer or
        can be converted to integer.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False

def list_integer_named_folders(root=".",prepend='',containing_files=[],not_containing_files=[]):
    """Return array of integer named folders.

    Scans folders in ``root`` and detects those which are named
    with an integer number. It returns then a sorted list of the
    found integers, e.g. , if the folder structure is::

        root/
            1/
            2/
            5/
            7/
            8/
            run_2/
            run_3/
            run_4/
            run_5/
            run_old/
            folderx/
            foldery/


    then the following array is returned::

        [1,2,5,7,8].

    If ``prepend`` is set to some string, then it returns a (sorted)
    list of strings. For example, if prepend is set to ``"run_"``, then this
    will return the array::

        ["run_2","run_3","run_4","run_5"]

    Parameters:

    root: string
        path of the root folder in which to scan for integer named folders.
    prepend: string
        scan for folders whose name starts with the string ``prepend`` and
        ends with an integer number.
    containing_files: array of strings
        a list of file names that should be contained in the returned folders.
    not_containing_files: array of strings
        a list of file names that should not be contained in the returned folders.

    Returns:
        array of integers if default value for ``prepend`` is used. Otherwise
        returns an array of strings

    """
    folders = sub_folders(root)

    flist = []
    for folder in folders:
        include = True
        for fn in not_containing_files:
            if os.path.exists(os.path.join(root,folder,fn)):
                include = False
                break
        if not include:
            continue

        for fn in containing_files:
            if not os.path.exists(os.path.join(root,folder,fn)):
                include = False
                break
        if not include:
            continue

        if prepend != '':
            if folder.startswith(prepend):
                d = folder[len(prepend):]
                if is_integer(d):
                    flist.append(int(d))
        else:
            if is_integer(folder):
                print("XrX",containing_files,folder)

                flist.append(int(folder))

    flist.sort()

    if prepend != '':
        slist = []
        for f in flist:
            slist.append(prepend + str(f))

        return slist

    else:
        return flist
