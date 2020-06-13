# make an ER graph
# N is the number of nodes
# p is the edge probability
# returns
#   k cumulative sum of degree vector
#   o list of edge endpoints. Thus, o[k[i-1]:k[i]] for i>0 is the set of neighbors of i
# additionally, these return values are written to an hdf5 file, together with the corresponding edge list
import numpy as np
import _make_er
def make_gnp(N,p):

    # choose the edge count
    m = np.random.binomial((N*(N-1))//2,p) # pick a random number of edges

    # generate random edges
    # generate extra in order to make m the expected number of unique edges
    # assumptions: N>5, p is small enough that triple edges, etc. are negligible, e.g. p=4/N
    AUGMENT = (1+p)*np.exp(-p) + .5 * (1 - (1+p)*np.exp(-p))
    e = np.random.randint(N*(N-1)//2,size=int(m/AUGMENT),dtype=np.uint64)
    e.sort()
    e = np.unique(e)
    m = e.size 
    
    # get edge coordinates
    (r,c) = _make_er.getrc(e)
    e.dtype=np.uint32

    # compute k
    k = np.bincount(e,minlength=N).astype(np.uint32)
    np.cumsum(k,out=k)

    # write k to file
    import h5py
    with h5py.File("gnp.h5",'w') as f:
        f.create_dataset("offsets",data=k)

    # write e to file (needed for next step)
    with h5py.File("gnp.h5",'a') as f:
        f.create_dataset("edge_list",data=e[:2*m],chunks=True)
    o=e # steal e's memory
    del r,c # their memory is about to be corrupted

    # compute o using k and e (the edge list)
    # we read e in chunks,
    with h5py.File("gnp.h5",'a') as f:
        e = f["edge_list"]

        # use k to write o to e's memory
        # so, k is used here as an array of offsets for writing elements of e into o
        # the elements of k are decremented each time the corresponding number is found in e
        # we will fix k later
        ch_size = int(1e8)
        print(int(m/ch_size))
        for ch_cnt in range(int(np.ceil(m/ch_size))):
            print(ch_cnt)
            ch = np.s_[2*ch_cnt*ch_size : min(2*(ch_cnt+1)*ch_size,2*m)]
            _make_er.make_oo(e[ch],o,k)
        f.create_dataset("endpoint_list",data=o)
    del e
    
    # get k back
    del k
    with h5py.File("gnp.h5",'a') as f:
        k = f['offsets'][:]

    return (k,o)

def _test_symmetry(k,o):
    _make_er.test_symmetry(k,o)
