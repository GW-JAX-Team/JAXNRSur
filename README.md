# JAXNRSur

### A JAX-based package for differentiable numerical relativity surrogate waveform generation

[![docs](https://img.shields.io/badge/docs-online-blue)](https://gw-jax-team.github.io/JAXNRSur/) [![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/GW-JAX-Team/JAXNRSur/blob/main/LICENSE) [![coverage](https://img.shields.io/coveralls/github/GW-JAX-Team/JAXNRSur/main)](https://coveralls.io/github/GW-JAX-Team/JAXNRSur?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/JAXNRSur/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/JAXNRSur/main)

JAXNRSur is a JAX-based package for differentiable evaluation of numerical relativity surrogate waveforms. By reimplementing the surrogate pipeline in JAX, it delivers NR-faithful time-domain waveforms for precessing and high-mass-ratio binary black holes with native GPU support and automatic differentiation — enabling gradient-based inference within modern gravitational-wave pipelines such as [Jim](https://github.com/GW-JAX-Team/jim).

**Supported models:**

- NRHybSur3dq8 — aligned-spin hybrid surrogate (q ≤ 8)
- NRSur7dq4 — precessing NR surrogate (q ≤ 4)

For a quick introduction, see the [Quick Start guide](https://gw-jax-team.github.io/JAXNRSur/stable/quickstart/).

## Installation

The simplest way to install JAXNRSur is through pip:

```bash
pip install JAXNRSur
```

This will install the latest stable release and its dependencies.
JAXNRSur is built on [JAX](https://github.com/jax-ml/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install JAXNRSur[cuda]
```

If you want to install the latest version of JAXNRSur, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/JAXNRSur.git
cd JAXNRSur
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.

## Attribution

If you use JAXNRSur in your research, please cite the underlying surrogate models:

- NRHybSur3dq8: [Varma et al. 2019](https://arxiv.org/abs/1812.07865)
- NRSur7dq4: [Varma et al. 2019](https://arxiv.org/abs/1905.09300)
