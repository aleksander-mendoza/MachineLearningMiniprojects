import numpy as np
import random
import torch


def norm(v):
    return v / np.sqrt((v ** 2).sum())


graph = [
    [1],  # 0
    [0, 1],  # 1
    [3],  # 2
    [],  # 3
]
DIM = len(graph)
graph_mat = torch.zeros(DIM, DIM)
for src, outgoing in enumerate(graph):
    for dst in outgoing:
        graph_mat[dst, src] += 1
column_sums = graph_mat.sum(dim=0)
for column_idx, column_sum in enumerate(column_sums):
    if column_sum == 0:
        for row in range(DIM):
            graph_mat[row, column_idx] = 1
        column_sums[column_idx] = DIM
likelihoods = torch.ones(DIM)
graph_mat.div_(column_sums.unsqueeze(0))

eig = graph_mat.eig(eigenvectors=True)
if (eig.eigenvalues[:, 1] != 0).all():
    print("Warning! You randomly got a matrix with complex eigenvalues")
print(eig)
for eig_vec_idx, eig_val in enumerate(eig.eigenvalues):
    if eig_val[0] == 1 and eig_val[1] == 0:
        eig_vec = eig.eigenvectors[:, eig_vec_idx]
        break

for iteration in range(0, 100):
    likelihoods = graph_mat @ likelihoods
likelihoods = norm(likelihoods)

eig_vec = norm(eig_vec)
print("approximation=", likelihoods)
print("eigen vector=", eig_vec)

current_page = random.randint(0, DIM - 1)
monte_carlo = np.zeros(DIM)
for iteration in range(0, 500):
    monte_carlo[current_page] += 1
    outgoing = graph[current_page]
    if len(outgoing) == 0:
        current_page = random.randint(0, DIM - 1)
    else:
        current_page = outgoing[random.randint(0, len(outgoing) - 1)]

monte_carlo = norm(monte_carlo)
print("monte carlo=", monte_carlo)
print("approximation / monte carlo ratio=", likelihoods / monte_carlo)
print("eigen vector / approximation ratio=", eig_vec / likelihoods)
print("monte carlo / eigen vector ratio=", monte_carlo / eig_vec)

# torch.return_types.eig(
# eigenvalues=tensor([[-0.5000,  0.0000],
#         [ 1.0000,  0.0000],
#         [-0.3904,  0.0000],
#         [ 0.6404,  0.0000]]),
# eigenvectors=tensor([[-0.7071, -0.4472,  0.3813,  0.2571],
#         [ 0.7071, -0.8944, -0.5955,  0.6587],
#         [ 0.0000,  0.0000, -0.3813, -0.2571],
#         [ 0.0000,  0.0000,  0.5955, -0.6587]]))
# approximation= tensor([4.4721e-01, 8.9443e-01, 9.1873e-21, 2.3534e-20])
# eigen vector= tensor([-0.4472, -0.8944,  0.0000,  0.0000])
# monte carlo= [0.46987975 0.88272208 0.00271607 0.00271607]
# approximation / monte carlo ratio= tensor([9.5176e-01, 1.0133e+00, 3.3826e-18, 8.6646e-18], dtype=torch.float64)
# eigen vector / approximation ratio= tensor([-1.0000, -1.0000,  0.0000,  0.0000])
# monte carlo / eigen vector ratio= tensor([-1.0507, -0.9869,     inf,     inf], dtype=torch.float64)
