"""
Solving Poisson's equation on the sphere
========================================

Poisson's equation relates a scalar field to a known source through the spherical Laplacian.
In harmonic space, the spherical Laplacian becomes a simple multiplication, so the equation can
be solved one coefficient at a time.

This tutorial uses ``S2FFT`` to solve Poisson's equation on the unit sphere and compares the
numerical solution with a known exact solution.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :align: center
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/astro-informatics/s2fft/tree/gh-pages/_colab_notebooks/spherical_poisson_equation.ipynb

If you are working on this notebook in Google Colab, you will need to have Google Colab install
``cartopy`` and ``s2fft``. You can do this by adding a cell to the top of the notebook with the
following content:

.. code-block:: bash

    !pip install cartopy s2fft &> /dev/null

and then running that cell.
"""

# %% [markdown]
# ## Poisson's equation on the sphere
#
# Consider an unknown scalar field $u(\theta,\phi)$ on the unit sphere, where $\theta \in [0,\pi]$
# is colatitude and $\phi \in [0,2\pi)$ is longitude.
# [Poisson's equation](https://en.wikipedia.org/wiki/Poisson%27s_equation) is
#
# $$
# \Delta_{\mathbb{S}^2} u = f,
# $$
#
# where $f(\theta,\phi)$ is a known source and $\Delta_{\mathbb{S}^2}$ is the [Laplace–Beltrami operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator),
# the spherical counterpart of the usual Laplacian. In spherical coordinates,
#
# $$
# \Delta_{\mathbb{S}^2} u
# = \frac{1}{\sin\theta} \frac{\partial}{\partial\theta} \left(\sin\theta \frac{\partial u}{\partial\theta}\right)
# + \frac{1}{\sin^{2}\theta} \frac{\partial^{2}u}{\partial\phi^{2}}.
# $$

# %% [markdown]
# ## Solving in harmonic space
#
# Spherical harmonics are eigenfunctions of the Laplace–Beltrami operator:
#
# $$
# \Delta_{\mathbb{S}^2} Y_{\ell m} = -\ell (\ell+1) Y_{\ell m}.
# $$
#
# Expanding $u$ and $f$ in spherical harmonics therefore turns Poisson's equation into a separate
# algebraic equation for each harmonic coefficient:
#
# $$
# -\ell (\ell+1) u_{\ell m} = f_{\ell m}.
# $$
#
# For every coefficient with $\ell \geq 1$, the solution is
#
# $$
# u_{\ell m} = -\frac{f_{\ell m}}{\ell(\ell+1)}.
# $$
#
# The constant mode $\ell = 0$ has eigenvalue zero because the Laplacian of a constant is zero.
# A solution therefore exists only if $f_{00}=0$, meaning that the source has zero mean. Since adding
# any constant to $u$ gives another solution, we select the zero-mean solution by setting $u_{00} = 0$.

# %%
import jax

jax.config.update("jax_enable_x64", True)

import cartopy.crs as ccrs
import numpy as np
from matplotlib import pyplot as plt

import s2fft

L = 32
sampling = "mw"
