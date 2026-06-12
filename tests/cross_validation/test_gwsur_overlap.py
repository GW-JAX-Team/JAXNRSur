"""Cross-validation: JAXNRSur waveforms vs gwsurrogate reference.

Compares the time-domain strain from JAXNRSur against gwsurrogate for both
NRSur7dq4 (precessing) and NRHybSur3dq8 (aligned-spin) over randomly sampled
parameters within each surrogate's validity range.

Both surrogates are evaluated on the SAME time grid with phi_ref=0 at
coalescence so there is no phase-reference ambiguity:

  NRSur7dq4:    gwsurrogate called with f_low=0 (returns ~2 s for typical
                masses); JAXNRSur called on the returned time array.
  NRHybSur3dq8: gwsurrogate would need ~10^9 samples with f_low=0 due to the
                PN hybrid start at ~-5.4e8 M.  Instead we pass a fixed
                `times` array (last `segment_length` seconds before merger)
                directly to gwsurrogate, keeping phi_ref at coalescence.

Phi conventions (empirically verified with f_low=0):
  NRSur7dq4:    gwsurrogate phi_ref=0 <-> JAXNRSur phi = 3*pi/2
  NRHybSur3dq8: gwsurrogate phi_ref=0 <-> JAXNRSur phi = 3*pi/2

Comparison region: samples where both |h_jax| and |h_gwsur| > 1% of peak.

Requires: gwsurrogate installed and surrogate data files present.
Run with: .venv/bin/pytest tests/cross_validation/ --n-samples=3
"""

import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

try:
    import gwsurrogate

    GWSUR_AVAILABLE = True
except ImportError:
    GWSUR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DT = 1.0 / 4096  # time step (s)
_SAMPLE_TIMEOUT_S = 120  # per-sample wall-clock timeout (seconds)

_NRSUR7DQ4_CFG = {
    "name": "NRSur7dq4",
    "M_range": [40.0, 100.0],
    "q_range": [1.0, 4.0],
    "chi_range": [0.0, 0.75],
    "d_L": 500.0,
    "phi_obs": 3.0 * np.pi / 2,
    "precessing": True,
    # f_low=0 is practical: NRSur7dq4 starts at ~-4500 M (~2 s for typical M)
    "use_times": False,
    "f_low": 0.0,
    "segment_length": None,
    # Thresholds set ~5 orders above achieved machine precision (~1e-15 amp_std).
    "amp_std_threshold": 1e-10,
    "amp_mean_tol": 1e-6,
    "phase_mean_threshold": 1e-6,  # degrees
    "phase_std_threshold": 1e-6,  # degrees
    "max_abs_err_threshold": 1e-10,
}

_NRHYBSUR3DQ8_CFG = {
    "name": "NRHybSur3dq8",
    "M_range": [40.0, 100.0],
    "q_range": [1.0, 8.0],
    "chi_range": [-0.75, 0.75],
    "d_L": 500.0,
    "phi_obs": 3.0 * np.pi / 2,
    "precessing": False,
    # Pass a fixed times array: avoids the ~10^9-sample PN start while keeping
    # phi_ref=0 at coalescence.
    # phi_obs=3pi/2: gwsurrogate projects at (inclination, pi/2-phi_ref)=(iota,pi/2);
    # JAXNRSur evaluates harmonics at (theta,-phi_obs)=(iota,-3pi/2)=(iota,pi/2). Match.
    "use_times": True,
    "f_low": 0.0,
    "segment_length": 4.0,  # seconds of waveform ending at merger
    # Thresholds set ~3 orders above achieved precision (~5e-9 amp_std).
    "amp_std_threshold": 1e-6,
    "amp_mean_tol": 1e-4,
    "phase_mean_threshold": 1e-3,  # degrees
    "phase_std_threshold": 1e-3,  # degrees
    "max_abs_err_threshold": 1e-5,
}


# ---------------------------------------------------------------------------
# Timeout helper (Unix SIGALRM)
# ---------------------------------------------------------------------------


class _SampleTimeout(Exception):
    pass


def _arm_timeout(seconds: int) -> None:
    def _handler(signum, frame):
        raise _SampleTimeout(f"timed out after {seconds}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)


def _disarm_timeout() -> None:
    signal.alarm(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_params(n: int, cfg: dict, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    M_min, M_max = cfg["M_range"]
    q_min, q_max = cfg["q_range"]
    chi_min, chi_max = cfg["chi_range"]

    params_list = []
    for _ in range(n):
        M_tot = rng.uniform(M_min, M_max)
        q = rng.uniform(q_min, q_max)
        iota = rng.uniform(0.1, np.pi - 0.1)

        if cfg["precessing"]:
            chi1_mag = rng.uniform(0.0, chi_max)
            chi2_mag = rng.uniform(0.0, chi_max)
            theta1 = np.arccos(rng.uniform(-1.0, 1.0))
            theta2 = np.arccos(rng.uniform(-1.0, 1.0))
            phi1 = rng.uniform(0.0, 2.0 * np.pi)
            phi2 = rng.uniform(0.0, 2.0 * np.pi)
            chiA = np.array(
                [
                    chi1_mag * np.sin(theta1) * np.cos(phi1),
                    chi1_mag * np.sin(theta1) * np.sin(phi1),
                    chi1_mag * np.cos(theta1),
                ]
            )
            chiB = np.array(
                [
                    chi2_mag * np.sin(theta2) * np.cos(phi2),
                    chi2_mag * np.sin(theta2) * np.sin(phi2),
                    chi2_mag * np.cos(theta2),
                ]
            )
        else:
            chi1z = rng.uniform(chi_min, chi_max)
            chi2z = rng.uniform(chi_min, chi_max)
            chiA = np.array([0.0, 0.0, chi1z])
            chiB = np.array([0.0, 0.0, chi2z])

        params_list.append(
            {"M_tot": M_tot, "q": q, "chiA": chiA, "chiB": chiB, "iota": iota}
        )

    return params_list


def _gwsur_waveform(sur, p: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate gwsurrogate; return (t_s, h) where h = h+ - i*hx in SI."""
    d_L = cfg["d_L"]

    if cfg["use_times"]:
        # Fixed time array ending at merger — avoids huge PN-hybrid arrays and
        # keeps phi_ref=0 at coalescence regardless of segment length.
        # f_low=0 is required by the API (no default); with `times` provided,
        # gwsurrogate does NOT generate the large PN dt grid (warning only
        # applies to dt-based calls).
        N = int(cfg["segment_length"] / _DT)
        t_si = np.arange(N) * _DT - cfg["segment_length"]
        t, h, _ = sur(
            p["q"],
            p["chiA"],
            p["chiB"],
            M=p["M_tot"],
            dist_mpc=d_L,
            times=t_si,
            f_low=0.0,
            units="mks",
            inclination=p["iota"],
            phi_ref=0.0,
        )
    else:
        t, h, _ = sur(
            p["q"],
            p["chiA"],
            p["chiB"],
            M=p["M_tot"],
            dist_mpc=d_L,
            f_low=cfg["f_low"],
            dt=_DT,
            units="mks",
            inclination=p["iota"],
            phi_ref=0.0,
        )

    return np.asarray(t), np.asarray(h)


def _build_jaxnrsur_wrapper(model_name: str):
    """Load the JAXNRSur model once — reused across all samples."""
    from jaxnrsur import JAXNRSur

    if model_name == "NRSur7dq4":
        from jaxnrsur.NRSur7dq4 import NRSur7dq4Model

        model = NRSur7dq4Model()
    else:
        from jaxnrsur.NRHybSur3dq8 import NRHybSur3dq8Model

        model = NRHybSur3dq8Model()

    return JAXNRSur(model, alpha_window=0.0)


def _build_params_full(p: dict, cfg: dict) -> jnp.ndarray:
    """Build the full JAXNRSur parameter vector [M, dL, iota, phi, model_params]."""
    import equinox as eqx

    if cfg["precessing"]:
        model_params = [
            p["q"],
            p["chiA"][0],
            p["chiA"][1],
            p["chiA"][2],
            p["chiB"][0],
            p["chiB"][1],
            p["chiB"][2],
        ]
    else:
        model_params = [p["q"], p["chiA"][2], p["chiB"][2]]
    return jnp.array([p["M_tot"], cfg["d_L"], p["iota"], cfg["phi_obs"], *model_params])


def _compare(h_jax: np.ndarray, h_gws: np.ndarray) -> dict:
    peak = np.max(np.abs(h_gws))
    thr = 0.01 * peak
    mask = (np.abs(h_gws) > thr) & (np.abs(h_jax) > thr)
    if mask.sum() < 10:
        return {"ok": False, "reason": f"too few comparison samples ({mask.sum()})"}

    ratio = np.abs(h_jax[mask]) / np.abs(h_gws[mask])
    phase_deg = np.angle(h_jax[mask] * np.conj(h_gws[mask])) * 180.0 / np.pi
    max_abs_err = float(np.max(np.abs(h_jax[mask] - h_gws[mask])) / peak)

    return {
        "ok": True,
        "n_compare": int(mask.sum()),
        "amp_mean": float(ratio.mean()),
        "amp_std": float(ratio.std()),
        "phase_mean": float(phase_deg.mean()),
        "phase_std": float(phase_deg.std()),
        "max_abs_err": max_abs_err,
    }


# ---------------------------------------------------------------------------
# Parallel gwsurrogate worker
# ---------------------------------------------------------------------------


def _gwsur_worker_fn(args: tuple) -> tuple:
    """Evaluate gwsurrogate for one sample in a worker process.

    Must be module-level (not a closure) so ProcessPoolExecutor can pickle it.
    Returns (t, h, error_str); error_str is None on success.
    """
    model_name, p, cfg = args
    try:
        import gwsurrogate as _gws

        sur = _gws.LoadSurrogate(model_name)
        t, h = _gwsur_waveform(sur, p, cfg)
        return t, h, None
    except Exception as exc:
        return None, None, str(exc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GWSUR_AVAILABLE, reason="gwsurrogate required")
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(_NRSUR7DQ4_CFG, id="NRSur7dq4"),
        pytest.param(_NRHYBSUR3DQ8_CFG, id="NRHybSur3dq8"),
    ],
)
def test_gwsur_td_agreement(cfg, n_samples, workers, cross_val_results):
    """JAXNRSur TD strain must agree with gwsurrogate to within thresholds."""
    model_name = cfg["name"]
    params_list = _generate_params(n_samples, cfg, seed=42)

    # --- Phase 1: gwsurrogate waveforms (parallel across CPU workers) ---
    worker_args = [(model_name, p, cfg) for p in params_list]
    if workers > 1:
        print(
            f"\n  [{model_name}] gwsurrogate: {n_samples} samples,"
            f" {workers} parallel workers ...",
            flush=True,
        )
        # spawn avoids fork() into JAX's multithreaded state (which causes deadlocks)
        _spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=_spawn_ctx) as pool:
            gwsur_outputs = list(pool.map(_gwsur_worker_fn, worker_args))
    else:
        sur = gwsurrogate.LoadSurrogate(model_name)
        print(
            f"\n  [{model_name}] gwsurrogate: {n_samples} samples, sequential ...",
            flush=True,
        )
        gwsur_outputs = []
        for p in params_list:
            try:
                t, h = _gwsur_waveform(sur, p, cfg)
                gwsur_outputs.append((t, h, None))
            except Exception as exc:
                gwsur_outputs.append((None, None, str(exc)))

    # --- Phase 2: JAXNRSur evaluation + comparison ---
    import equinox as eqx

    wrapper = _build_jaxnrsur_wrapper(model_name)

    amp_means, amp_stds, phase_means, phase_stds, max_abs_errs = [], [], [], [], []
    per_sample: list[dict] = []
    failed: list[tuple[int, str]] = []

    # Mark gwsur failures up front so they don't enter the JAXNRSur paths.
    gws_failed = {i for i, (_, _, err) in enumerate(gwsur_outputs) if err is not None}
    for i in gws_failed:
        print(
            f"  [{i + 1:2d}/{n_samples}] M={params_list[i]['M_tot']:.1f}"
            f" q={params_list[i]['q']:.2f}  GWSUR FAILED: {gwsur_outputs[i][2]}"
        )
        failed.append((i, f"gwsurrogate: {gwsur_outputs[i][2]}"))

    valid_idx = [i for i in range(n_samples) if i not in gws_failed]

    if cfg["use_times"] and valid_idx:
        # NRHybSur3dq8: all samples share the same fixed time array.
        # Ripple pattern: jit(vmap) over the full parameter batch, then compare.
        t_fixed = jnp.array(gwsur_outputs[valid_idx[0]][0])

        params_batch = jnp.stack(
            [_build_params_full(params_list[i], cfg) for i in valid_idx]
        )

        def _eval_single(p_full):
            hp, hc = wrapper.get_waveform_td(t_fixed, p_full, f_lower=0.0)
            return hp - 1j * hc

        def _is_oom(e: Exception) -> bool:
            msg = str(e)
            return "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg

        def _run_batch(batch_size: int | None):
            if batch_size is None:
                fn = eqx.filter_jit(jax.vmap(_eval_single))
                return np.array(fn(params_batch))
            fn = eqx.filter_jit(
                lambda xs: jax.lax.map(_eval_single, xs, batch_size=batch_size)
            )
            return np.array(fn(params_batch))

        batch_size: int | None = None
        n_valid = len(valid_idx)
        print(
            f"\n  [{model_name}] JAXNRSur: compiling+evaluating"
            f" {n_valid} samples with jit+vmap ...",
            flush=True,
        )
        while True:
            try:
                h_jax_batch = _run_batch(batch_size)
                break
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                next_bs = max(1, (n_valid if batch_size is None else batch_size) // 2)
                if batch_size is not None and next_bs == batch_size:
                    # Even batch_size=1 OOMs; fall back to per-sample JIT loop.
                    print(
                        f"\n  [OOM] falling back to per-sample jit loop ...",
                        flush=True,
                    )
                    _fn_single = eqx.filter_jit(_eval_single)
                    h_jax_batch = np.stack(
                        [np.array(_fn_single(params_batch[k])) for k in range(n_valid)]
                    )
                    break
                print(
                    f"\n  [OOM] retrying with lax.map(batch_size={next_bs}) ...",
                    flush=True,
                )
                batch_size = next_bs

        for k, i in enumerate(valid_idx):
            p = params_list[i]
            print(
                f"  [{i + 1:2d}/{n_samples}] M={p['M_tot']:.1f} q={p['q']:.2f}",
                end="",
                flush=True,
            )
            result = _compare(h_jax_batch[k], gwsur_outputs[i][1])
            if not result["ok"]:
                print(f"  SKIP: {result['reason']}")
                failed.append((i, result["reason"]))
                continue
            amp_means.append(result["amp_mean"])
            amp_stds.append(result["amp_std"])
            phase_means.append(result["phase_mean"])
            phase_stds.append(result["phase_std"])
            max_abs_errs.append(result["max_abs_err"])
            per_sample.append({**p, **result})
            print(
                f"  amp={result['amp_mean']:.6f}±{result['amp_std']:.2e}"
                f"  ph={result['phase_mean']:+.4f}°±{result['phase_std']:.4f}°"
                f"  max_err={result['max_abs_err']:.2e}"
                f"  n={result['n_compare']}"
            )

    else:
        # NRSur7dq4 (or fallback): variable-length time arrays per sample.
        # JIT-compile once; JAX recompiles only when time-array shape changes.
        _fn = eqx.filter_jit(lambda t, p: wrapper.get_waveform_td(t, p, f_lower=0.0))
        for i in valid_idx:
            p = params_list[i]
            t_gws, h_gws, _ = gwsur_outputs[i]
            print(
                f"  [{i + 1:2d}/{n_samples}] M={p['M_tot']:.1f} q={p['q']:.2f} ...",
                end="",
                flush=True,
            )
            try:
                _arm_timeout(_SAMPLE_TIMEOUT_S)
                params_full = _build_params_full(p, cfg)
                hp, hc = _fn(jnp.array(t_gws), params_full)
                h_jax = np.array(hp) - 1j * np.array(hc)
                _disarm_timeout()

                result = _compare(h_jax, h_gws)
                if not result["ok"]:
                    print(f"  SKIP: {result['reason']}")
                    failed.append((i, result["reason"]))
                    continue

                amp_means.append(result["amp_mean"])
                amp_stds.append(result["amp_std"])
                phase_means.append(result["phase_mean"])
                phase_stds.append(result["phase_std"])
                max_abs_errs.append(result["max_abs_err"])
                per_sample.append({**p, **result})
                print(
                    f"  amp={result['amp_mean']:.6f}±{result['amp_std']:.2e}"
                    f"  ph={result['phase_mean']:+.4f}°±{result['phase_std']:.4f}°"
                    f"  max_err={result['max_abs_err']:.2e}"
                    f"  n={result['n_compare']}"
                )
            except _SampleTimeout as exc:
                _disarm_timeout()
                print(f"  TIMEOUT: {exc}")
                failed.append((i, str(exc)))
            except Exception as exc:
                _disarm_timeout()
                print(f"  FAILED: {exc}")
                failed.append((i, str(exc)))

    n_ok = len(amp_means)
    if n_ok == 0:
        pytest.fail(f"{model_name}: all {n_samples} samples failed/timed out")

    overall_amp_mean = float(np.mean(amp_means))
    overall_amp_std = float(np.mean(amp_stds))
    overall_phase_mean = float(np.mean(np.abs(phase_means)))
    overall_phase_std = float(np.mean(phase_stds))
    overall_max_abs_err = float(np.max(max_abs_errs))

    print(f"\n{model_name} vs gwsurrogate (TD, {n_ok}/{n_samples} ok):")
    print(f"  amp mean: {overall_amp_mean:.6f}  amp std (avg): {overall_amp_std:.2e}")
    print(
        f"  |phase mean| avg: {overall_phase_mean:.6f}°  phase std avg: {overall_phase_std:.6f}°"
    )
    print(f"  max |h_jax - h_gws| / peak (worst sample): {overall_max_abs_err:.2e}")

    amp_std_thr = cfg["amp_std_threshold"]
    amp_mean_tol = cfg["amp_mean_tol"]
    phase_mean_thr = cfg["phase_mean_threshold"]
    phase_std_thr = cfg["phase_std_threshold"]
    max_abs_err_thr = cfg["max_abs_err_threshold"]

    results_dir = Path(__file__).parent / "results" / f"n{n_samples}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"td_comparison_{model_name}.csv"
    with open(results_file, "w") as f:
        f.write(
            "M_tot,q,iota,"
            "chiA_x,chiA_y,chiA_z,chiB_x,chiB_y,chiB_z,"
            "amp_mean,amp_std,phase_mean_deg,phase_std_deg,max_abs_err\n"
        )
        for s in per_sample:
            cA, cB = s["chiA"], s["chiB"]
            f.write(
                f"{s['M_tot']:.4f},{s['q']:.4f},{s['iota']:.4f},"
                f"{cA[0]:.6f},{cA[1]:.6f},{cA[2]:.6f},"
                f"{cB[0]:.6f},{cB[1]:.6f},{cB[2]:.6f},"
                f"{s['amp_mean']:.8f},{s['amp_std']:.6e},"
                f"{s['phase_mean']:.6f},{s['phase_std']:.6f},{s['max_abs_err']:.6e}\n"
            )
    print(f"  Results saved to: {results_file}")

    passed = (
        len(failed) == 0
        and abs(overall_amp_mean - 1.0) < amp_mean_tol
        and overall_amp_std < amp_std_thr
        and overall_phase_mean < phase_mean_thr
        and overall_phase_std < phase_std_thr
        and overall_max_abs_err < max_abs_err_thr
    )

    cross_val_results.append(
        {
            "model": model_name,
            "n_samples": n_samples,
            "n_finite": n_ok,
            "n_failed": len(failed),
            "amp_mean": overall_amp_mean,
            "amp_std": overall_amp_std,
            "phase_mean": overall_phase_mean,
            "phase_std": overall_phase_std,
            "max_abs_err": overall_max_abs_err,
            "amp_threshold": amp_std_thr,
            "phase_threshold": phase_std_thr,
            "passed": passed,
        }
    )

    assert len(failed) == 0, (
        f"{model_name}: {len(failed)}/{n_samples} samples failed/timed out\n"
        + "\n".join(f"  [{i}] {msg}" for i, msg in failed)
    )
    assert abs(overall_amp_mean - 1.0) < amp_mean_tol, (
        f"{model_name}: amp mean {overall_amp_mean:.6f} off from 1.0 by >{amp_mean_tol}"
    )
    assert overall_amp_std < amp_std_thr, (
        f"{model_name}: amp ratio std {overall_amp_std:.2e} > {amp_std_thr:.0e}"
    )
    assert overall_phase_mean < phase_mean_thr, (
        f"{model_name}: |phase mean| {overall_phase_mean:.6f}° > {phase_mean_thr}°"
    )
    assert overall_max_abs_err < max_abs_err_thr, (
        f"{model_name}: max |h_jax-h_gws|/peak {overall_max_abs_err:.2e} > {max_abs_err_thr:.0e}"
    )
    assert overall_phase_std < phase_std_thr, (
        f"{model_name}: phase std {overall_phase_std:.6f}° > {phase_std_thr}°"
    )
