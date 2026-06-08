# Installation

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

## Data cache

Surrogate data files are downloaded automatically from Zenodo on first use and cached in `$XDG_CACHE_HOME/.JAXNRSur` (defaults to `~/.cache/.JAXNRSur` when `XDG_CACHE_HOME` is not set). On HPC systems where `XDG_CACHE_HOME` points to a project or scratch space, the data will be stored there automatically.
