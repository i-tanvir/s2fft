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
# On the unit sphere, Poisson's equation for an unknown scalar field $u(\theta,\phi)$ is
#
# $$
# \nabla_{\mathbb{S}^2}^{2} u = f,
# $$
#
# where $f(\theta,\phi)$ is a known source and $\nabla_{\mathbb{S}^2}^{2}$ is the
# [Laplace-Beltrami operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator).
# In spherical coordinates,
#
# $$
# \nabla_{\mathbb{S}^2}^{2} u
# = \frac{1}{\sin\theta} \frac{\partial}{\partial\theta} \left(\sin\theta \frac{\partial u}{\partial\theta}\right)
# + \frac{1}{\sin^{2}\theta} \frac{\partial^{2}u}{\partial\phi^{2}}.
# $$

