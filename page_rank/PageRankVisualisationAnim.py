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
        plt.title("Iteration "+str(iteration))
        plt.grid(True)
        plt.axis("equal")
        for eig_vec in eig.eigenvectors.T:
            plt.arrow(0, 0, eig_vec[0], eig_vec[1])

        intermediate_transformation = j / steps * transformation + (1 - j / steps) * np.eye(2)
        transformed_likelihood = np.dot(intermediate_transformation, likelihood_numpy)
        plt.arrow(0, 0, transformed_likelihood[0], transformed_likelihood[1], color=(1, 0, 0))
        transgrid = np.dot(intermediate_transformation, xygrid)  # apply intermediate matrix transformation
        plt.scatter(transgrid[0], transgrid[1], s=10, edgecolor="none", color=(0, 0, 1))
        plt.pause(interval=min(0.001, duration / steps))

    likelihoods = graph @ likelihoods
    print(iteration, likelihoods / likelihoods.sum())
print(eig)
print(eig.eigenvectors[:, 0], '==', graph @ eig.eigenvectors[:, 0])
print(likelihoods.sum(), eig.eigenvectors[:, 0].sum(), likelihoods / eig.eigenvectors[:, 0])
