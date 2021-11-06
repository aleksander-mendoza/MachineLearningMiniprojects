import numpy as np
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt


def norm(v):
    return v / np.sqrt((v ** 2).sum())


graph = [
    [1, 2],  # 0
    [0, 1],  # 1
    [3, 4],  # 2
    [],  # 3
    [],  # 4
]
DIM = len(graph)
graph_mat = torch.zeros(DIM, DIM)
for src, outgoing in enumerate(graph):
    for dst in outgoing:
        graph_mat[dst, src] += 1
column_sums = graph_mat.sum(dim=0)

graph_mat_verticies = graph_mat.numpy().copy()

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
    if abs(eig_val[0].item()-1)< 0.00001 and abs(eig_val[1].item())< 0.00001:
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

    A = np.matrix(graph_mat_verticies)

    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    pos = nx.random_layout(G, seed=4423)
    color_map = []

    for node in G:
        if node == current_page:
            color_map.append('#ffed99')
        else:
            color_map.append('#f6b8b8')

    nx.draw(G, node_color=color_map, pos=pos)
    plt.pause(0.001)
    plt.clf()

print(graph_mat_verticies)

monte_carlo = norm(monte_carlo)
print("monte carlo=", monte_carlo)
print("approximation / monte carlo ratio=", likelihoods / monte_carlo)
print("eigen vector / approximation ratio=", eig_vec / likelihoods)
print("monte carlo / eigen vector ratio=", monte_carlo / eig_vec)

# torch.return_types.eig(
# eigenvalues=tensor([[ 1.0000,  0.0000],
#         [-0.2804,  0.1728],
#         [-0.2804, -0.1728],
#         [ 0.4609,  0.0000],
#         [ 0.0000,  0.0000]]),
# eigenvectors=tensor([[ 4.6657e-01,  5.5362e-03, -3.1222e-01,  3.2265e-01, -2.1220e-09],
#         [ 6.9985e-01, -2.5021e-01,  8.6901e-02,  6.7269e-01, -2.2814e-08],
#         [ 3.4993e-01, -6.4278e-01,  0.0000e+00, -5.7111e-02, -4.0679e-08],
#         [ 2.9161e-01,  4.4373e-01,  1.1266e-01, -4.6911e-01, -7.0711e-01],
#         [ 2.9161e-01,  4.4373e-01,  1.1266e-01, -4.6911e-01,  7.0711e-01]]))
# approximation= tensor([0.4666, 0.6999, 0.3499, 0.2916, 0.2916])
# eigen vector= tensor([0.4666, 0.6999, 0.3499, 0.2916, 0.2916])
# [[0. 1. 0. 0. 0.]
#  [1. 1. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0.]]
# monte carlo= [0.45967742 0.75806452 0.2983871  0.25       0.25      ]
# approximation / monte carlo ratio= tensor([1.0150, 0.9232, 1.1727, 1.1664, 1.1664], dtype=torch.float64)
# eigen vector / approximation ratio= tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
# monte carlo / eigen vector ratio= tensor([0.9852, 1.0832, 0.8527, 0.8573, 0.8573], dtype=torch.float64)
