import numpy as np
import matplotlib.pyplot as plt

def plot_classification_boundaries(x, y, features, model_obj, alpha=.3):
    """contour plot of the boundaries of the classification object model"""
    min1, max1 = x[features[0]].min() - 1, x[features[1]].max() + 1
    min2, max2 = x[features[0]].min() - 1, x[features[1]].max() + 1

    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))
    yhat = model_obj.predict(grid)

    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class
    for value in y.unique():
        # create scatter of these obs
        plt.scatter(x[y == value][features[0]], x[y == value][features[1]], cmap='Paired', alpha=alpha)
