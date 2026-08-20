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
# $Y_{lm}(\theta,\phi) = N_{lm} \cdot P_{l}^m(cos\theta) \cdot exp(im\phi)$
#
# where $P_{l}^m$ is an associated Legendre function and $N_{lm}$ is a normalisation constant.
# Spherical harmonics are usually complex-valued because of the term $exp(im\phi)$.
# %%
