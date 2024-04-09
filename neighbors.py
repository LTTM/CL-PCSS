"""
mask = (np.vectorize(label_to_step.__getitem__)(
np.arange(0, len(label_to_step)))) != self.step  # sequential masked
"""


import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
from utils.SemanticKITTIdataset import SemanticKITTIDataset

dataset = SemanticKITTIDataset(augment=False, split='val')

num_classes = 20
ov_mat = np.zeros((num_classes,num_classes))

from joblib import Parallel, delayed

@staticmethod
def process_data(X, Y):
    X = X.squeeze(0)

    nbrs = NearestNeighbors(n_neighbors=16, algorithm='ball_tree', n_jobs=64).fit(X)
    distances, indices = nbrs.kneighbors(X)
    #gpu_resource = faiss.StandardGpuResources()  # use a single GPU
    #cpu_index = faiss.IndexFlatL2()  # create a CPU index
    #gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)  # transfer the index to GPU
    #gpu_index.add(X)  # add vectors to the index
    #distances, indices = gpu_index.search(X, 16)

    YY = Y[indices]
    classes = np.unique(Y)

    nn_mat = np.zeros((num_classes,num_classes))
    dist_mat = np.zeros((num_classes,num_classes))

    for n,j in enumerate(YY):
        i = Y[n]
        nn_mat[i,j] += 1
        dist_mat[i,j] = ((dist_mat[i,j] * (nn_mat[i,j]-1)) + distances[n,:]) / nn_mat[i,j] #running mean

    print("-")
    return nn_mat, dist_mat

ov_mat = np.zeros((num_classes,num_classes))
d_mat = np.zeros((num_classes,num_classes))
results = Parallel(n_jobs=2)(delayed(process_data)(X, Y) for X, Y in dataset)

for count, (nn_mat, dist_mat) in enumerate(results):
    ov_mat += nn_mat
    d_mat = ((d_mat * (count)) + dist_mat) / (count+1)
np.save("mat_nn2.npy", ov_mat)
np.save("mat_d2.npy", d_mat)

np.save("results.npy", results)

#normalize ov_mat
nov_mat = np.zeros_like(ov_mat)
for i in range(0, 20):
    for j in range(0, 20):
        nov_mat[i, j] = ov_mat[i, j] / sum(ov_mat[i, :])

for i in range(0, 20):
    for j in range(0, 20):
        if d_mat[i,j] == 0:
            d_mat[i,j] = np.Inf
        else:
            d_mat[i, j] = d_mat[i, j] / sum(d_mat[i, :])

A_mat = np.multiply(nov_mat, 1/d_mat)
print(A_mat)
np.save("A_mat", A_mat)