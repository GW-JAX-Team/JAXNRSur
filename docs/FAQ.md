# FAQ

## Float precision

JAX defaults to float32, but surrogate waveform computations require float64 precision. Without it, you may see inaccurate waveforms or unexpected NaN values. Always enable it at the top of your script, **before** any JAX operations:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## JIT compilation time

The first call to a JIT-compiled model triggers XLA compilation, which can take several seconds. This is normal — subsequent calls will be much faster. If you are timing JAXNRSur for benchmarking purposes, discard the first call.

To disable JIT for debugging:

```python
jax.config.update("jax_disable_jit", True)
```

## Compilation is slow for complex pipelines

If you wrap a JAXNRSur model inside a larger likelihood with many operations or Python-level loops, JAX may take a long time to compile the full computational graph. Replacing Python loops with `jax.lax.scan` or `jax.vmap` where possible can significantly reduce compilation time.

## Surrogate data download

On first use, JAXNRSur downloads the surrogate data files from Zenodo and caches them in `$XDG_CACHE_HOME/.JAXNRSur` (defaults to `~/.cache/.JAXNRSur`). The download only happens once. On HPC systems, set `XDG_CACHE_HOME` to a project or scratch directory with sufficient storage before running.
