import copy

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)
n = 8
colors = np.random.rand(n)
points = np.random.rand(n, 2)
clusters = [[p] for p in range(n)]

plt.show()


def single_link(c1, c2):
    min_diameter = 9999999
    min_point0, min_point1 = None, None
    for p1 in c1:
        for p2 in c2:
            dist = np.linalg.norm(points[p1] - points[p2])
            if dist < min_diameter:
                min_diameter = dist
                min_point0, min_point1 = points[p1], points[p2]
    return min_diameter, min_point0, min_point1


def complete_link(c1, c2):
    max_diameter = -1
    max_point0, max_point1 = None, None
    for p1 in c1:
        for p2 in c2:
            dist = np.linalg.norm(points[p1] - points[p2])
            if dist > max_diameter:
                max_diameter = dist
                max_point0, max_point1 = points[p1], points[p2]
    return max_diameter, max_point0, max_point1


def average_link(c1, c2):
    avg_diameter = 0
    avg_point0, avg_point1 = np.array([0, 0], dtype=np.float), np.array([0, 0], dtype=np.float)
    for p1 in c1:
        for p2 in c2:
            avg_diameter += np.linalg.norm(points[p1] - points[p2])
            avg_point0 += points[p1]
            avg_point1 += points[p2]
    total = len(c1) * len(c2)
    avg_diameter /= total
    avg_point0 /= total
    avg_point1 /= total
    return avg_diameter, avg_point0, avg_point1


def centroids_link(c1, c2):
    avg_point0, avg_point1 = np.array([0, 0], dtype=np.float), np.array([0, 0], dtype=np.float)
    for p1 in c1:
        for p2 in c2:
            avg_point0 += points[p1]
            avg_point1 += points[p2]
    total = len(c1) * len(c2)
    avg_point0 /= total
    avg_point1 /= total
    dist = np.linalg.norm(avg_point0 - avg_point1)
    return dist, avg_point0, avg_point1


def wards_method(c1, c2):
    mean = np.array([0, 0], dtype=float)
    for p1 in c1:
        mean += points[p1]
    for p2 in c2:
        mean += points[p2]
    total = len(c1) + len(c2)
    mean /= total
    variance = np.array([0, 0], dtype=float)
    for p1 in c1:
        variance += (mean - points[p1]) ** 2
    for p2 in c2:
        variance += (mean - points[p2]) ** 2
    variance = np.sqrt(variance[0] + variance[1])
    return variance, mean, mean


method = wards_method
if method == wards_method:
    title = "Ward's Method (minimal variance)"
elif method == centroids_link:
    title = "Centroid distance"
elif method == average_link:
    title = "Average distance between points"
elif method == complete_link:
    title = "Complete link (Longest point distance)"
elif method == single_link:
    title = "Single link (Shortest point distance)"
interval = 0.5


# min_x, max_x, min_y, max_y = points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()
# dist_x = max_x - min_x
# dist_y = max_y - min_y

def plt_setup():
    b = 0.1
    # Set the limits of the plot
    plt.xlim(0 - b, 1 + b)
    plt.ylim(0 - b, 1 + b)

    # Don't mess with the limits!
    plt.autoscale(False)
    plt.title(title)
    plt.annotate(' '.join(['[' + ', '.join([chr(ord('A') + i) for i in c]) + ']' for c in clusters]), (-b,-b))


while len(clusters) > 1:
    cluster_closest_to = [(999999, None, None) for _ in range(len(clusters))]

    def plot_variance(dist, end1, end2, c1, c2, circle_c, line_style):
        assert (end1 == end2).all()
        circle1 = plt.Circle(end1, dist, color=circle_c, alpha=0.2)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(circle1)
        ax.scatter(end1[0], end1[1], marker='x', c='black')
        for p1 in c1:
            plt.plot([end1[0], points[p1, 0]], [end1[1], points[p1, 1]], c=str(colors[p1]), linestyle=line_style)
        for p2 in c2:
            plt.plot([end1[0], points[p2, 0]], [end1[1], points[p2, 1]], c=str(colors[p2]), linestyle=line_style)

    def scatter_points(colors):
        plt.scatter(points[:, 0], points[:, 1], c=colors)
        a = ord('A')
        for i, point in enumerate(points):
            plt.annotate(chr(a + i), point)


    def plot_closest():
        for source1, (dist1, destination1, endpoint1) in enumerate(cluster_closest_to):
            if destination1 is not None:
                dist2, destination2, endpoint2 = cluster_closest_to[destination1]
                if destination2 == source1:  # found a valid link!
                    if (endpoint2 == endpoint1).all():
                        assert dist1 == dist2
                        c1, c2 = clusters[destination1], clusters[destination2]
                        plot_variance(dist2, endpoint1, endpoint2, c1, c2, 'red', ':')
                    else:
                        plt.plot([endpoint1[0], endpoint2[0]], [endpoint1[1], endpoint2[1]], c='red')


    for c1_idx in range(len(clusters)):
        for c2_idx in range(len(clusters)):
            if c1_idx != c2_idx:
                dist1, _, _ = cluster_closest_to[c1_idx]
                dist2, _, _ = cluster_closest_to[c2_idx]
                c1, c2 = clusters[c1_idx], clusters[c2_idx]
                dist, end1, end2 = method(c1, c2)
                if dist1 > dist:
                    cluster_closest_to[c1_idx] = (dist, c2_idx, end1)
                if dist2 > dist:
                    cluster_closest_to[c2_idx] = (dist, c1_idx, end2)

                plt.clf()
                scatter_points(colors)
                if (end1 == end2).all():
                    plot_variance(dist, end1, end2, c1, c2, 'yellow', '-')
                else:
                    plt.plot([end1[0], end2[0]], [end1[1], end2[1]])
                plot_closest()
                plt_setup()
                plt.pause(interval=interval)
    plt.clf()
    scatter_points(colors)
    plot_closest()
    plt_setup()
    plt.pause(interval=interval * 2)

    new_clusters = []
    old_clusters = copy.copy(clusters)
    new_colors = np.copy(colors)
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
                    new_colors[p2] = color
                new_clusters.append(c1 + c2)
                clusters[source1] = None
                clusters[destination1] = None
    for c in clusters:
        if c is not None:
            new_clusters.append(c)

    clusters = old_clusters
    for i in range(9):
        plt.clf()
        scatter_points(new_colors if i % 2 == 0 else colors)
        plot_closest()
        plt_setup()
        plt.pause(interval=1 / 8)

    clusters = new_clusters
    colors = new_colors

plt.show()
