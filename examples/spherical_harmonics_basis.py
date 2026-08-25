"""
Understanding spherical harmonics
=================================

Spherical harmonics are the spherical counterpart of Fourier modes.
Just as Fourier modes provide a basis for representing periodic functions,
spherical harmonics provide a basis for representing functions defined on
the sphere.

This tutorial introduces their main mathematical properties, visualises
some basis functions, and shows how ``S2FFT`` decomposes a signal into
spherical harmonic coefficients.
"""

# %% [markdown]
# ## Spherical coordinates
#
# A point on the unit sphere is described by two angles:
#
# * $\theta \in [0,\pi]$ is the colatitude, measured down from the north pole.
# * $\phi \in [0,2\pi)$ is the longitude around the sphere, measured eastward from a reference
# [meridian](https://en.wikipedia.org/wiki/Meridian_(geography)).
#
# We write a signal on the sphere as $f(\theta,\phi)$. Depending on the application,
# it could represent quantities such as temperature, elevation, or radiation intensity.

# %% [markdown]
# ## Spherical harmonics
#
# The spherical harmonic $Y_{\ell m}(\theta,\phi)$ is indexed by two integers:
#
# * The degree $\ell=0,1,2,\dots$ controls its overall angular scale.
# * The order $m=-\ell,\dots,\ell$ controls its variation with longitude.
#
# A spherical harmonic can be written as
#
# $$
# Y_{\ell m}(\theta,\phi)
# = N_{\ell m} P_{\ell}^m(\cos\theta) \exp(i m\phi),
# $$
#
# where $P_{\ell}^m$ is an [associated Legendre polynomial](https://en.wikipedia.org/wiki/Associated_Legendre_polynomials)
# and $N_{\ell m}$ is a normalisation constant. The factor $\exp(i m\phi)$ means
# that spherical harmonics are generally complex-valued.

# %% [markdown]
# ## Orthonormality
#
# The surface-area element on the unit sphere, also called the solid-angle element, is
#
# $$
# \mathrm{d}\Omega = \sin\theta \mathrm{d}\theta \mathrm{d}\phi.
# $$
#
# For two square-integrable functions $f$ and $g$, the inner product on the sphere is
#
# $$
# \langle f,g\rangle
# = \int_{\mathbb{S}^{2}}
# f(\theta,\phi) g^{*}(\theta,\phi) \ \mathrm{d}\Omega.
# $$
#
# Here, $*$ denotes complex conjugation. The factor $\sin\theta$ in
# $\mathrm{d}\Omega$ arises from the [solid-angle element in spherical coordinates](https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates).
#
# The spherical harmonics are orthonormal with respect to the inner product:
#
# $$
# \langle Y_{\ell m},Y_{\ell' m'}\rangle
# = \int_{\mathbb{S}^{2}}
# Y_{\ell m}(\theta,\phi) Y^{*}_{\ell' m'}(\theta,\phi) \ \mathrm{d}\Omega
# = \delta_{\ell \ell'} \delta_{m m'}.
# $$
#
# The Kronecker delta $\delta_{ab}$ equals one when $a=b$ and zero otherwise. Consequently,
# distinct spherical harmonics have zero inner product, while each spherical harmonic
# has unit inner product with itself.

# %% [markdown]
# ## Expanding a signal in spherical harmonics
#
# The spherical harmonics form a complete basis for $L^{2}(\mathbb{S}^{2})$,
# the space of [square-integrable functions](https://en.wikipedia.org/wiki/Square-integrable_function)
# on the sphere. Any signal in this space can therefore be expanded as
#
# $$
# f(\theta,\phi)
# = \sum_{\ell=0}^{\infty} \sum_{m=-\ell}^{\ell}
# f_{\ell m} Y_{\ell m}(\theta,\phi),
# $$
#
# where orthonormality allows each coefficient to be isolated by an inner product:
#
# $$
# f_{\ell m}
# = \langle f,Y_{\ell m}\rangle
# = \int_{\mathbb{S}^{2}}
# f(\theta,\phi) Y^{*}_{\ell m}(\theta,\phi) \ \mathrm{d}\Omega.
# $$
#
# Computing the coefficients $f_{\ell m}$ is called the forward spherical harmonic transform,
# or spherical harmonic analysis. Reconstructing $f$ from these coefficients is called the
# inverse spherical harmonic transform, or spherical harmonic synthesis.

# %% [markdown]
# ## Band-limited signals
#
# S2FFT computes transforms for signals with a finite harmonic band-limit $L$. A signal is
# band-limited at $L$ if $f_{\ell m}=0$ for every $\ell \geq L$. Its expansion then becomes
# the finite sum
#
# $$
# f(\theta,\phi)
# = \sum_{\ell=0}^{L-1} \sum_{m=-\ell}^{\ell}
# f_{\ell m} Y_{\ell m}(\theta,\phi).
# $$
#
# The largest degree represented is therefore $L-1$. Increasing $L$ allows finer angular
# structure to be represented, but also increases the number of coefficients and the
# computational cost of the transforms.
#
# There are
#
# $$
# \sum_{\ell=0}^{L-1}(2\ell +1)=L^{2}
# $$
#
# valid spherical harmonic coefficients.
# A sampling theorem specifies a finite set of sample locations from which a band-limited signal
# can be represented and transformed exactly, up to numerical precision. Here we use the [McEwen-Wiaux](https://arxiv.org/abs/1110.6298)
# sampling scheme.

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
# ## How S2FFT stores harmonic coefficients
#
# S2FFT stores coefficients in an array `flm` of shape `(L, 2*L-1}`. The first array
# index corresponds directly to the degree $\ell$. The order $m$ is shifted by $L-1$ so that
# its negative and positive values can be represented by standard non-negative array indices:
#
# $$
# f_{\ell m} \quad \longleftrightarrow \quad
# \mathtt{flm[\ell,\ L-1+m]}.
# $$
#
# Thus, $m=0$ is stored in the central column with index $L-1$. The rectangular array also contains
# entries for which $|m|>\ell$. These do not correspond to valid spherical harmonic coefficients
# and are kept at zero.

# %%
# Choose a valid degree and order: 0 <= ell < L, and -ell <= m <= ell.
ell = 2
m = 1

# Set all coefficients to zero, then set f_{ell m} = 1.
flm = np.zeros(s2fft.sampling.s2_samples.flm_shape(L), dtype=np.complex128)
flm[ell, L-1+m] = 1.0

# Since this is the only non-zero coefficient, the inverse transform evaluates
# Y_{ell m} at the MW sampling nodes.
y_ell_m = s2fft.inverse(
    flm,
    L=L,
    sampling=sampling,
    method="jax",
    reality=False,
)

# %% [markdown]
# Spherical harmonics are generally complex-valued, so here we visualise the
# real part.

# %%
fig, ax = plt.subplots(
    figsize=(4, 2),
    subplot_kw={"projection": ccrs.Mollweide()},
)

ax.imshow(
    y_ell_m.real,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
)
ax.set_title(rf"$\ell={ell},\ m={m}$")

plt.show()

# %% [markdown]
# ## Visualising the basis
#
# We can repeat the same construction, setting one coefficient to one and all
# remaining coefficients to zero, for every valid degree and order $(\ell,m)$
# up to a chosen maximum degree.

# %%
maximum_degree = 3

fig, axes = plt.subplots(
    maximum_degree + 1,
    2 * maximum_degree + 1,
    figsize=(12, 6),
    subplot_kw={"projection": ccrs.Mollweide()},
)

for ell in range(maximum_degree + 1):
    for m in range(-maximum_degree, maximum_degree + 1):
        ax = axes[ell, m + maximum_degree]

        # Hide positions that do not correspond to a valid order.
        if abs(m) > ell:
            ax.set_axis_off()
            continue

        flm = np.zeros(
            s2fft.sampling.s2_samples.flm_shape(L),
            dtype=np.complex128,
        )
        flm[ell, L - 1 + m] = 1.0

        basis_function = s2fft.inverse(
            flm,
            L=L,
            sampling=sampling,
            method="jax",
            reality=False,
        )

        ax.imshow(
            basis_function.real,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
        )
        ax.set_title(rf"$\ell={ell},\ m={m}$", fontsize=9)

plt.show()

# %% [markdown]
# Moving down the rows increases $\ell$, producing finer angular structure.
# Moving across a row changes $m$ and therefore the variation with longitude.
# When $m=0$, the harmonic does not vary with longitude.

# %% [markdown]
# Decomposing a signal into spherical harmonics
#
# A band-limited signal is a weighted sum of spherical harmonic basis functions.
# To demonstrate this, we construct a signal with three non-zero coefficients:
#
# $$
# f(\theta, \phi)
# = Y_{0,0}(\theta,\phi)
# + 0.75Y_{2,1}(\theta,\phi)
# - 0.5Y_{3,-2}(\theta,\phi).
# $$
# 
# The inverse transform, evaluates this weighted sum at the MW sampling nodes.

# %%
signal_flm = np.zeros(
    s2fft.sampling.s2_samples.flm_shape(L),
    dtype=np.complex128,
)
signal_flm[0, L - 1] = 1.0
signal_flm[2, L - 1 + 1] = 0.75
signal_flm[3, L - 1 - 2] = -0.5

signal = s2fft.inverse(
    signal_flm,
    L=L,
    sampling=sampling,
    method="jax",
    reality=False,
)

# %% [markdown]
# The signal is generally complex-valued because the spherical harmonic basis
# functions are complex-valued. As before, we visualise its real part.

# %%
fig, ax = plt.subplots(
    figsize=(6,4),
    subplot_kw={"projection": ccrs.Mollweide()},
)

ax.imshow(
    signal.real,
    transform=ccrs.PlateCarree(),
    cmap="viridis", 
)

plt.show()
