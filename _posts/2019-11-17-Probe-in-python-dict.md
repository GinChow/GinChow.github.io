---
layout: post
title: 谈谈Python中的Dict
author: Gin 
excerpt_separator: <!--more-->
toc: true
tag: Exploration
categories: [Python, C]
---

Have a look on Python dict

<!--more-->

## History of Python Dict

This is 2010 Pycon talk about python dict, which introduces the mechanism of Dict implementation.

[The Mighty Dict](https://www.youtube.com/watch?v=rWdF7oW6z18)

The implementation was deprecated on 2015(Python 3.4), for details, refer

[Pypy announce](https://morepypy.blogspot.com/2015/01/faster-more-memory-efficient-and-more.html)

[Raymond Hettinger's mail](https://mail.python.org/pipermail/python-dev/2012-December/123028.html)


## Data structure of Python Dict

Let's take a glance at the data structure.

```c

typedef struct {
    /* Cached hash code of me_key. */
    Py_hash_t me_hash;
    PyObject *me_key;
    PyObject *me_value; /* This field is only meaningful for combined tables */
} PyDictKeyEntry;

struct _dictkeysobject {
    Py_ssize_t dk_refcnt;

    /* Size of the hash table (dk_indices). It must be a power of 2. */
    Py_ssize_t dk_size;

    /* Function to lookup in the hash table (dk_indices):

       - lookdict(): general-purpose, and may return DKIX_ERROR if (and
         only if) a comparison raises an exception.

       - lookdict_unicode(): specialized to Unicode string keys, comparison of
         which can never raise an exception; that function can never return
         DKIX_ERROR.

       - lookdict_unicode_nodummy(): similar to lookdict_unicode() but further
         specialized for Unicode string keys that cannot be the <dummy> value.

       - lookdict_split(): Version of lookdict() for split tables. */
    dict_lookup_func dk_lookup;

    /* Number of usable entries in dk_entries. */
    Py_ssize_t dk_usable;

    /* Number of used entries in dk_entries. */
    Py_ssize_t dk_nentries;

    /* Actual hash table of dk_size entries. It holds indices in dk_entries,
       or DKIX_EMPTY(-1) or DKIX_DUMMY(-2).

       Indices must be: 0 <= indice < USABLE_FRACTION(dk_size).

       The size in bytes of an indice depends on dk_size:

       - 1 byte if dk_size <= 0xff (char*)
       - 2 bytes if dk_size <= 0xffff (int16_t*)
       - 4 bytes if dk_size <= 0xffffffff (int32_t*)
       - 8 bytes otherwise (int64_t*)

       Dynamically sized, SIZEOF_VOID_P is minimum. */
    char dk_indices[];  /* char is required to avoid strict aliasing. */

    /* "PyDictKeyEntry dk_entries[dk_usable];" array follows:
       see the DK_ENTRIES() macro */
};
```

The former implementation of Dict is just consisted by *PyDictKeyEntry* , whose layout looks like:

For example, the dictionary:

    d = {'timmy': 'red', 'barry': 'green', 'guido': 'blue'}

is stored as:

    entries = [['--', '--', '--'],
               [-8522787127447073495, 'barry', 'green'],
               ['--', '--', '--'],
               ['--', '--', '--'],
               ['--', '--', '--'],
               [-9092791511155847987, 'timmy', 'red'],
               ['--', '--', '--'],
               [-6480567542315338377, 'guido', 'blue']]

Apparently, the entries array is sparse and space wasting, because lots of slots is empty. Instead of using sparse key-value array, the latest implementation replace it with a sparse index array, which looks like 

    indices =  [None, 1, None, None, None, 0, None, 2]
    entries =  [[-9092791511155847987, 'timmy', 'red'],
                [-8522787127447073495, 'barry', 'green'],
                [-6480567542315338377, 'guido', 'blue']]
