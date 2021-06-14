import matplotlib.pyplot as plt
import numpy as np

n = 8
colors = np.random.rand(n)
points = np.random.rand(n, 2)
clusters = [[p] for p in range(n)]

plt.show()


def min_diameter(c1, c2):
    min_diameter = 9999999
    min_point0, min_point1 = None, None
    for p1 in c1:
        for p2 in c2:
            dist = np.linalg.norm(points[p1] - points[p2])
            if dist < min_diameter:
                min_diameter = dist
                min_point0, min_point1 = points[p1], points[p2]
    return min_diameter, min_point0, min_point1


method = min_diameter
interval = 0.001

while len(clusters) > 1:
    cluster_closest_to = [(999999, None, None) for _ in range(len(clusters))]

    def scatter_points():
        plt.scatter(points[:, 0], points[:, 1], c=colors)
        a = ord('A')
        for i, point in enumerate(points):
            plt.annotate(chr(a+i), point)

    def plot_closest():
        for source1, (_, destination1, endpoint1) in enumerate(cluster_closest_to):
            if destination1 is not None:
                _, destination2, endpoint2 = cluster_closest_to[destination1]
                if destination2 == source1:  # found a valid link!
                    plt.plot([endpoint1[0], endpoint2[0]], [endpoint1[1], endpoint2[1]], c='red')

    for c1_idx in range(len(clusters)):
        for c2_idx in range(len(clusters)):
            if c1_idx != c2_idx:
                dist1, _, _ = cluster_closest_to[c1_idx]
                dist2, _, _ = cluster_closest_to[c2_idx]
                c1, c2 = clusters[c1_idx], clusters[c2_idx]
                dist, end1, end2 = method(c1, c2)
                if dist1 > dist and dist2 > dist:
                    cluster_closest_to[c1_idx] = (dist, c2_idx, end1)
                    cluster_closest_to[c2_idx] = (dist, c1_idx, end2)
                scatter_points()
                plt.plot([end1[0], end2[0]], [end1[1], end2[1]])
                plot_closest()
                plt.pause(interval=interval)
                plt.clf()
    print({i: e for i, e in enumerate(cluster_closest_to)})
    scatter_points()
    plot_closest()
    plt.pause(interval=interval * 10)
    plt.clf()
    new_clusters = []
    for source1, (_, destination1, endpoint1) in enumerate(cluster_closest_to):
        if destination1 is not None:
            _, destination2, endpoint2 = cluster_closest_to[destination1]
            if destination2 == source1:  # found a valid link!
                c1 = clusters[source1]
                c2 = clusters[destination1]
                if c1 is None and c2 is None:
                    continue  # already merged
                representative1 = c1[0]
                color = colors[representative1]
                for p2 in c2:
                    colors[p2] = color
                new_clusters.append(c1 + c2)
                clusters[source1] = None
                clusters[destination1] = None
    for c in clusters:
        if c is not None:
            new_clusters.append(c)
    scatter_points()
    plot_closest()
    plt.pause(interval=interval*10)
    plt.clf()
    clusters = new_clusters
