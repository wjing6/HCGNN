from scipy.sparse import csr_matrix
import numpy as np
def get_csr_from_coo(src, dst):
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (src, dst)))
    return csr_mat

print (get_csr_from_coo(np.array([1,2,3]), np.array([0,1,2])))
