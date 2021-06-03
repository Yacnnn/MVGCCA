import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def imshow_grid(images, shape=[2, 2], title = ""):
    """Plot images in a grid of a given shape. Images input size : [batch,X,Y,channel]"""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.
    plt.title("uci")
    plt.show()
