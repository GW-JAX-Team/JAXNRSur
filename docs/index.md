# JAXNRSur

## A JAX-based package for differentiable numerical relativity surrogate waveform generation

JAXNRSur is a JAX-based package for differentiable evaluation of numerical relativity surrogate waveforms. By reimplementing the surrogate pipeline in JAX, it delivers NR-faithful time-domain waveforms for precessing and high-mass-ratio binary black holes with native GPU support and automatic differentiation — enabling gradient-based inference within modern gravitational-wave pipelines such as [Jim](https://github.com/GW-JAX-Team/jim).

**Supported models:**

- NRHybSur3dq8 — aligned-spin hybrid surrogate (q ≤ 8)
- NRSur7dq4 — precessing NR surrogate (q ≤ 4)

## Documentation

- **[Installation](installation.md)** — How to install JAXNRSur
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Step-by-step guides and worked examples
- **[FAQ](FAQ.md)** — Answers to common questions
- **[Contributing](contributing.md)** — How to contribute to JAXNRSur
