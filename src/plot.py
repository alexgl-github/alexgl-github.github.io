import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-10, 10, 0.1)
Y = np.arange(-10, 10, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.exp(X) / (np.exp(X) + np.exp(Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.1, 1.1)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel(r'$x_{1}$', fontsize=20)
ax.set_ylabel(r'$x_{2}$', fontsize=20)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\sigma(x_{1})$', fontsize=20, rotation=0, labelpad=20)

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
