import numpy as np
from matplotlib import pyplot as plt
import torch

graph = torch.rand(2, 2)
likelihoods = torch.ones(2)
graph.div_(graph.sum(dim=0).unsqueeze(0))

eig = graph.eig(eigenvectors=True)
if (eig.eigenvalues[:, 1] != 0).all():
    print("Warning! You randomly got a matrix with complex eigenvalues")
print(eig)
for eig_vec_idx, eig_val in enumerate(eig.eigenvalues):
    if eig_val[0] == 1 and eig_val[1] == 0:
        eig_vec = eig.eigenvectors[:, eig_vec_idx]
        break

xvals = np.linspace(-4, 4, 9)
yvals = np.linspace(-3, 3, 7)
xygrid = np.column_stack([[x, y] for x in xvals for y in yvals])
transformation = graph.numpy()

steps = 60
duration = 1

for iteration in range(0, 100):
    likelihood_numpy = likelihoods.numpy()
    for j in range(steps + 1):
        plt.clf()
        plt.title("Iteration " + str(iteration))
        plt.grid(True)
        plt.axis("equal")
        for ev in eig.eigenvectors.T:
            plt.arrow(0, 0, ev[0], ev[1])

        intermediate_transformation = j / steps * transformation + (1 - j / steps) * np.eye(2)
        transformed_likelihood = np.dot(intermediate_transformation, likelihood_numpy)
        plt.arrow(0, 0, transformed_likelihood[0], transformed_likelihood[1], color=(1, 0, 0))
        transgrid = np.dot(intermediate_transformation, xygrid)  # apply intermediate matrix transformation
        plt.scatter(transgrid[0], transgrid[1], s=10, edgecolor="none", color=(0, 0, 1))
        plt.pause(interval=min(0.001, duration / steps))

    likelihoods = graph @ likelihoods
    print("iteration=", iteration)
    print("eigen vector=", eig_vec)
    print("approximation=", likelihoods)
    print("eigen vector/approximation ratio=", likelihoods / eig_vec)

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

