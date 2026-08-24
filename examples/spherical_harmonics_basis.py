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
# * The degree $\ell=0,1,2,\dots$ controls its overall angular scale. Larger values of $\ell$
# correspond to finer angular structure.
# * The order $m=-\ell,\dots,\ell$ controls its variation with longitude. Larger values of ∣m∣
# correspond to more rapid longitudinal oscillations.
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
# ## Orthonormality and completeness
#
# For two square-integrable functions $f$ and $g$, the inner product on the sphere is
# 
# $$
# \langle f,g\rangle
# = \int_{0}^{2\pi}\int_{0}^{\pi}
# f(\theta,\phi) g^{*}(\theta,\phi) \sin\theta \ \mathrm{d}\theta \mathrm{d}\phi.
# $$
#
# Here, $*$ denotes complex conjugation. The factor $\sin\theta$ arises from the 
# [solid-angle element in spherical coordinates](https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates):
# 
# $$
# \mathrm{d}\Omega = \sin\theta \mathrm{d}\theta \mathrm{d}\phi.
# $$
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
#
# The spherical harmonics also form a complete basis for $L^{2}(\mathbb{S}^{2})$,
# the space of square-integrable functions on the sphere. Any signal in this space can therefore
# be expanded as
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
# 











# %%