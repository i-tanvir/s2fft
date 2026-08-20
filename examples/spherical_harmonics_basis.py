"""
Understanding spherical harmonics
=================================

Spherical harmonics are the spherical counterpart of Fourier waves.
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
# * $l=0,1,2,\dots$ is the degree, which controls its overall spatial frequency.
# * $m=-l,\dots,l$ is the order, which controls its variation with longitude.
#
# Generally, a spherical harmonic has the form
#
# $Y_{lm}(\theta,\phi) = N_{lm} \ P_{l}^m(cos\theta) \ exp(im\phi)$
#
# where $P_{l}^m$ is an associated Legendre function and $N_{lm}$ is a normalisation constant.
# Spherical harmonics are usually complex-valued because of the term $exp(im\phi)$.
#
# An important property of spherical harmonics is orthonormality
#
# $ \langle Y_{lm},Y_{l'm'} \rangle = \int_{0}^{2\pi} \int_{0}^{\pi} Y_{lm}(\theta,\phi) \ Y^*_{l',m'}(\theta,\phi) \ sin\theta \ \mathrm{d}\theta = \delta_{ll'} \ \delta_{mm'}$
#
# where $*$ means complex conjugate, and $\delta$ is the Kronecker delta,
# which is 1 when its two indices match, and 0 otherwise. In other words,
# distint spherical harmonics have zero inner product, while every basis function
# has unit inner product with itself.
#
# Because the spherical harmonics form a complete orthonormal basis for square-integrable
# functions on the sphere, such a function can be expanded as
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

import jax # Import JAX before S2FFT so that we can enable 64-bit precision

jax.config.update("jax_enable_x64", True) # Spherical harmonic transforms are numerically much more accurate in 64-bit mode

import cartopy.crs as ccrs # Cartopy supplies map projections for displaying data sampled on a sphere.

import numpy as np # NumPy is used to create and inspect arrays of harmonic coefficients.

import s2fft # S2FFT supplies the forward and inverse spherical-harmonic transforms.

from matplotlib import pyplot as plt # # Matplotlib supplies the plotting functions used throughout the tutorial

sampling = "mw" # Use the McEwen--Wiaux sampling theorem

L = 32 # The band-limit L means that degrees l=0,...,L-1 can be represented.

