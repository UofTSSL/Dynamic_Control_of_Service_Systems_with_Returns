import numpy as np
import pickle
from matplotlib import pyplot as plt

pickle_files = ['knn_linear.pkl', 'knn_piecewise.pkl', 'knn_quadratic.pkl']
knns = []
for file in pickle_files:
    with open(file, 'rb') as f:
        knns.append(pickle.load(f))

x_dim = 561
y_dim = 561
x_grid = np.linspace(0, 140, x_dim)
y_grid = np.linspace(0, 140, y_dim)
xy_grid = np.array([[(x, y) for y in y_grid] for x in x_grid])

inputs = xy_grid.reshape(-1, 2)
preds = [knn.predict(inputs) for knn in knns]
outputs = [pred.reshape(x_dim, y_dim) for pred in preds]



for i, output in enumerate(outputs[-1:]):
    fig, ax = plt.subplots()
    img = ax.pcolor(x_grid, y_grid, output.T, vmin=0.1, vmax=0.2)
    plt.show()
    
    ax.set_xlabel(r'Needy State ($x$)')
    ax.set_ylabel(r'Content State ($y$)')
    ax.set_title('Optimal Policy ($p^*$)')
    ax.plot(np.linspace(0, 50), np.linspace(45, 45), linestyle='dotted')
    ax.axvline(50, linestyle='dotted')

    fig.set_figheight(3)
    fig.set_figwidth(3)
    
    # plt.savefig("policy_{}.png".format(i), dpi=600, bbox_inches='tight')

fig, ax = plt.subplots()
fig.set_figheight(3)
fig.set_figwidth(3)
ax.set_visible(False)
plt.colorbar(img, orientation='horizontal')
plt.savefig("colorbar.png", dpi=600, bbox_inches='tight')

if True:
    
    fig, axes = plt.subplots(1, len(outputs), sharey=True)
    imgs = [ax.pcolor(x_grid, y_grid, output.T, vmin=0.1, vmax=0.2) for ax, output in zip(axes, outputs)]

    
    axes[0].set_ylabel(r'Content State ($y$)')
    
    for ax in axes:
        ax.set_xlabel(r'Needy State ($x$)')
        ax.set_title('Optimal Policy ($p^*$)')
        ax.plot(np.linspace(0, 50), np.linspace(45, 45), linestyle='dotted')
        ax.axvline(50, linestyle='dotted')
    
    fig.colorbar(imgs[0], ax=axes)
    
    fig.set_figheight(3)
    fig.set_figwidth(11)
    # plt.tight_layout()
    
    plt.savefig('policy_combined.png', dpi=600, bbox_inches='tight')
