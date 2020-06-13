# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=0 CYTHON_TRACE=1
# cython: language_level=3, boundscheck=False
# cython: language_level=3, wraparound=False
# cython: language_level=3, binding=True
# cython: language_level=3, cdivision=True
# cython: infer_types=True
# distutils: extra_compile_args=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION -Wno-unused-variable -Wno-unused-function -Wno-unused-result -fopenmp
# distutils: extra_link_args=-fopenmp

import numpy as np
cimport numpy as np
DTYPE=np.uint64
ctypedef np.uint16_t DTYPE16_t
ctypedef np.uint32_t DTYPE32_t
ctypedef np.uint64_t DTYPE_t
ctypedef long double DTYPE128_t
FLOAT=np.float64
ctypedef np.float64_t FLOAT_t
cimport cython
from cython.parallel import prange
from libc.math cimport log
from libc.stdlib cimport malloc,free

def getrc(np.ndarray[DTYPE_t,ndim=1] e):
    my_union = np.dtype({ 'names': ['e','r','c'], 'formats': ['u8','u4','u4'],'offsets': [0,0,4]}) 
    e.dtype=my_union
    cdef:
        int m = e.size
        int i
        DTYPE_t r # need extra precision for the squaring and the square root
        DTYPE32_t c
        DTYPE_t [:]   ev = e['e']
        DTYPE32_t [:] rv = e['r']
        DTYPE32_t [:] cv = e['c']
    for i in prange(m,nogil=True):
        r = <DTYPE_t>(((1+8*<DTYPE128_t>ev[i])**.5 + 1)/2) # cast to double for the square root has loss of precision -- watch out!
        c = ev[i] - <DTYPE_t>(r*(r-1)//2) # potential problems with machine epsilon -- watch out! 
        rv[i] = <DTYPE32_t>(r)
        cv[i] = c
    return (e['r'],e['c'])

def make_oo(DTYPE32_t [:] e, DTYPE32_t [::1] o,DTYPE32_t [::1] k):
    cdef int i
    cdef int m=e.size/2
    #for i in prange(m,nogil=True): # may not be safe
    for i in range(m):
        k[e[2*i]] -= 1
        o[k[e[2*i]]] = e[2*i+1]
        k[e[2*i+1]] -= 1
        o[k[e[2*i+1]]] = e[2*i]

def test_symmetry(DTYPE32_t [::1] k, DTYPE32_t [::1] o):
    cdef int N = k.size
    cdef int i,j,found,iind,jind
    cdef int errcount=0

    #i=0 is special
    e = o[0:k[0]]
    if e.size > 0:
        for j in np.nditer(e):
            if j>0:
                assert(0 in o[k[j-1]:k[j]])

    for i in prange(1,N,nogil=True):
        for jind in range(k[i-1],k[i]):        
            j = o[jind]

            found=0;                            
            for iind in range(k[j-1],k[j]):     
                found = found + (o[iind] == i)
            errcount += (found == 0);

    if errcount == 0:
        print('test passed!')
    else:
        print('errors found')
