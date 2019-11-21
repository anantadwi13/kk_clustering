import os
import KMeans
import matplotlib.pyplot as plt


if __name__ == '__main__':

    selected_attrib = (0, 1)

    kmeans = KMeans.Algorithm(os.path.dirname(os.path.abspath(__file__)) + '\\dataset\\iris.data', 4, 4, 5, selected_attrib_idx=selected_attrib)
    res = kmeans.train()

    c = []
    x = []
    y = []

    x_cent = []
    y_cent = []
    c_cent = []

    for idx_cluster, cluster in enumerate(res):
        for member in cluster.members:
            c.append(idx_cluster)
            x.append(member.pos[selected_attrib[0]])
            y.append(member.pos[selected_attrib[1]])

        c_cent.append(idx_cluster)
        x_cent.append(cluster.pos[selected_attrib[0]])
        y_cent.append(cluster.pos[selected_attrib[1]])

    plt.scatter(x, y, c=c, marker='o', cmap='gist_rainbow')
    plt.scatter(x_cent, y_cent, c='black', marker=(5, 1), cmap='gist_rainbow')

    plt.xlabel('Sepal Length', fontsize=18)
    plt.ylabel('Sepal Width', fontsize=18)
    plt.show()
