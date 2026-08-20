"""
Understanding spherical harmonics
=================================

Spherical harmonics are the spherical counterpart of Fourier waves.
Just as Fourier modes provide a basis for representing periodic functions,
spherical harmonics provide a basis for representing functions defined on
the sphere.
This tutorial introduces their main mathematical properties, visualises
some basis functions, and shows how ''S2FFT'' decomposes a signal into
spherical harmonic coefficients
"""

# %% [markdown]
# Spherical coordinates and spherical harmonics
# ---------------------------------------------
#
# A point on the unit sphere is described by two angles:
#
# * $\theta \in [0,\pi]$ is the colatitude, measured down from the north pole.
# * $\phi \in [0,2\pi)$ is the longitude around the sphere.
#
# The spherical harmonic $Y_{lm}(\theta,\phi)$ is labelled by two integers:
#
# * $l=0,1,2,\dots$ is the degree, which controls its overall angular scale/spatial frequency.
# * $m=-l,\dots,l$ is the order, which controls its variation with longitude.
#
# A spherical harmonic can be written in the form
#
# $Y_{lm}(\theta,\phi) = N_{lm} \ P_{l}^m(\cos\theta) \ \exp(im\phi)$
#
# where $P_{l}^m$ is an associated Legendre function and $N_{lm}$ is a normalisation constant.
# Spherical harmonics are usually complex-valued because of the factor $exp(im\phi)$.
#
# An important property of spherical harmonics is orthonormality
#
# $ \langle Y_{lm},Y_{l'm'} \rangle = \int_{0}^{2\pi} \int_{0}^{\pi} Y_{lm}(\theta,\phi) \ Y^*_{l',m'}(\theta,\phi) \ \sin\theta \ \mathrm{d}\theta \ \mathrm{d}\phi = \delta_{ll'} \ \delta_{mm'}$
#
# where $*$ means complex conjugate, and $\delta$ is the Kronecker delta,
# which equals 1 when its two indices match, and 0 otherwise. In other words,
# distinct spherical harmonics have zero inner product, while each spherical harmonic
# has unit inner product with itself.
#
# Spherical harmonics form a complete orthonormal basis for square-integrable functions on the sphere.
# Therefore, a function $f(\theta,\phi) can be represented as a sum of spherical harmonics,
#
# $f(\theta,\phi) = \sum_{l=0}^{\infty} \sum_{m=-l}^{l} f_{lm} \ Y_{lm}(\theta,\phi)$
# 
# with coefficients
# 
# $f_{lm} = \langle f,Y_{lm} \rangle = \int_{\mathbb{S}^2} f(\theta,\phi) \ Y_{lm}^{*}(\theta,\phi) \ \mathrm{d}\Omega$
#
# S2FFT computes the coefficients $f_{lm}$ with a forward transform and reconstructs $f$ from
# those coefficients with an inverse transform. We use a finite band-limit $L$, so only
# degrees $0\leq l < L$ are represented.
#

# Import JAX before S2FFT so that we can enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import s2fft
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

# Use the McEwen-Wiaux sampling with band-limit L = 32
sampling = "mw"
L = 32


# %%
