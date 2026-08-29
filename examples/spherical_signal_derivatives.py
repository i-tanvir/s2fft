"""
Spatial derivatives of spherical signals
========================================

Spherical harmonic coefficients provide an accurate way to differentiate band-limited
signals on the sphere. Instead of approximating a derivative from neighbouring samples,
we differentiate the spherical harmonic expansion and then transform the result back
to the sampling grid.

This tutorial computes derivatives with respect to colatitude and longitude and compares
them with the exact derivatives of a known test signal.
"""

# %% [markdown]
# ## Coordinate derivatives on the sphere
#
# A scalar signal on the unit sphere is written as $f(\theta,\phi)$, where
# $\theta \in [0,\pi]$ is colatitude and $\phi \in [0,2\pi)$ is longitude.
# Its surface gradient is
#
# $$
# \nabla_{\mathbb{S}^2} f
# = \boldsymbol{e}_{\theta} \frac{\partial f}{\partial \theta}
# + \boldsymbol{e}_{\phi} \frac{1}{\sin \theta} \frac{\partial f}{\partial \phi},
# $$
#
# where $\boldsymbol{e}*{\theta}$ and $\boldsymbol{e}*{\phi}$ are unit vectors in the
# colatitude and longitude directions, respectively. The factor $1/\sin\theta$
# accounts for the variation in physical distance associated with a change in
# longitude across the sphere.
#
# We first compute the two coordinate derivatives $\partial f/\partial\theta$ and
# $\partial f/\partial\phi$. These derivatives are also fundamental building blocks for
# [spectral methods](https://en.wikipedia.org/wiki/Spectral_method) used to solve partial
# differential equations on the sphere.


# %%
import jax

jax.config.update("jax_enable_x64", True)

import cartopy.crs as ccrs
import numpy as np
from matplotlib import pyplot as plt

import s2fft

L = 32
sampling = "mw"

# %% [markdown]
# ## A band-limited signal
#
# We use the real-valued signal
#
# $$
# f(\theta,\phi)
# = \sin\theta\cos\phi
# + \frac{1}{2}\left(3\cos^{2}\theta-1\right).
# $$
#
# The first and second terms contain spherical harmonic modes of degrees one and two,
# respectively. The signal therefore contains no harmonic content above degree two and
# is band-limited. Its exact coordinate derivatives are:
#
# $$
# \frac{\partial f}{\partial\theta}
# = \cos\theta\cos\phi-3\sin\theta\cos\theta,
# $$
#
# $$
# \frac{\partial f}{\partial\phi}
# = -\sin\theta\sin\phi.
# $$
#

# %%
theta_values = s2fft.sampling.s2_samples.thetas(L, sampling)
phi_values = s2fft.sampling.s2_samples.phis_equiang(L, sampling)
theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing="ij")
longitude_grid = np.rad2deg(phi_grid)
latitude_grid = 90 - np.rad2deg(theta_grid)

signal = np.sin(theta_grid) * np.cos(phi_grid) + 0.5 * (3 * np.cos(theta_grid) ** 2 - 1)

exact_colatitude_derivative = (np.cos(theta_grid) * np.cos(phi_grid) - 3 * np.sin(theta_grid) * np.cos(theta_grid))
exact_longitude_derivative = -np.sin(theta_grid) * np.sin(phi_grid)

# %% [markdown]
# ## Visualising the signal
#
# We first visualise the sampled signal that will be differentiated.

# %%
fig, ax = plt.subplots(
    figsize=(6, 3),
    subplot_kw={"projection": ccrs.Mollweide()},
)

ax.pcolormesh(
    longitude_grid, latitude_grid,
    signal,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
)

ax.set_title(r"$f(\theta,\phi)$")
fig.tight_layout()

plt.show()

# %% [markdown]
# ## Transforming to harmonic space
#
# The forward transform decomposes the sampled signal into spherical harmonic coefficients
# $f_{\ell m}$, which are used to compute the coordinate derivatives in harmonic space.

# %%
flm = np.asarray(
    s2fft.forward(
        signal,
        L=L,
        sampling=sampling,
        method="jax",
        reality=True,
    )
)

# %% [markdown]
# ## Differentiating with respect to longitude
#
# The longitude dependence of a spherical harmonic is $\exp(i m\phi)$.
# Therefore,
#
# $$
# \frac{\partial}{\partial\phi}Y_{\ell m}(\theta,\phi)
# = i m Y_{\ell m}(\theta,\phi).
# $$
#
# Hence, the longitude derivative of the signal can be computed directly in harmonic space by
# multiplying each coefficient of order $m$ by $i m$. The inverse transform then evaluates the
# derivative at the sampling nodes.

# %%
m_values = np.arange(-L + 1, L)
longitude_derivative_flm = 1j * m_values * flm

longitude_derivative = np.asarray(
    s2fft.inverse(
        longitude_derivative_flm,
        L=L,
        sampling=sampling,
        method="jax",
        reality=True,
    )
)