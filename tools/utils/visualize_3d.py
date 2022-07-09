
import matplotlib.pyplot as plt


def plot_3d(x, y, lbls_co=None, figsize=(6, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    for k in set(y):
        label = lbls_co[k] if lbls_co is not None else k
        ax.scatter(x[:, 0][y == k],
                   x[:, 1][y == k],
                   x[:, 2][y == k],
                   label=label)
    plt.legend()
    plt.show()
