"""Cross-validation test configuration and session summary.

Collects per-model amplitude/phase statistics from all test_gwsur_* runs and
prints a formatted summary at the end of the session, including hardware info
and pass/fail status per model.
"""

import json
import platform
from datetime import datetime
from itertools import groupby
from pathlib import Path

import jax
import pytest


# ---------------------------------------------------------------------------
# Session-level results store
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config._cross_val_results = []


def pytest_addoption(parser):
    parser.addoption(
        "--n-samples",
        type=int,
        default=10,
        help="Number of random parameter sets per model (default: 10)",
    )
    parser.addoption(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for gwsurrogate evaluation (default: 1, use CPU count on GPU nodes)",
    )


@pytest.fixture(scope="session")
def n_samples(request):
    return request.config.getoption("--n-samples")


@pytest.fixture(scope="session")
def workers(request):
    return request.config.getoption("--workers")


@pytest.fixture(scope="session")
def cross_val_results(request):
    """Session-scoped list that accumulates per-model result dicts.

    Each entry has the shape::

        {
            "model": str,
            "n_samples": int,
            "n_finite": int,
            "n_failed": int,
            "amp_mean": float,   # mean(|h_jax| / |h_gws|) over in-band samples
            "amp_std": float,    # std of amplitude ratio
            "phase_mean": float, # mean phase difference (degrees)
            "phase_std": float,  # std of phase difference (degrees)
            "amp_threshold": float,
            "phase_threshold": float,
            "passed": bool,
        }
    """
    return request.config._cross_val_results


# ---------------------------------------------------------------------------
# Terminal summary hook
# ---------------------------------------------------------------------------


def _hardware_info() -> dict:
    import jax.numpy as jnp

    devices = jax.devices()
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "jax_devices": [str(d) for d in devices],
        "jax_version": jax.__version__,
        "float_dtype": str(jnp.zeros(1).dtype),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    results = getattr(config, "_cross_val_results", [])
    if not results:
        return

    hw = _hardware_info()

    terminalreporter.write_sep(
        "=", "Cross-Validation Summary (JAXNRSur vs gwsurrogate)"
    )

    terminalreporter.write_line("Hardware / Runtime")
    terminalreporter.write_line(f"  Host       : {hw['host']}")
    terminalreporter.write_line(f"  OS         : {hw['os']}")
    terminalreporter.write_line(f"  CPU        : {hw['cpu']}")
    terminalreporter.write_line(f"  Python     : {hw['python']}")
    terminalreporter.write_line(f"  JAX        : {hw['jax_version']}")
    terminalreporter.write_line(f"  Devices    : {', '.join(hw['jax_devices'])}")
    terminalreporter.write_line(
        f"  Precision  : {hw['float_dtype']} (x64_enabled={hw['x64_enabled']})"
    )
    terminalreporter.write_line(f"  Timestamp  : {hw['timestamp']}")
    terminalreporter.write_line("")

    col_w = 20
    num_w = 11
    header = (
        f"{'Model':<{col_w}}"
        f"{'N':>{num_w}}"
        f"{'Failed':>{num_w}}"
        f"{'AmpMean':>{num_w}}"
        f"{'AmpStd':>{num_w}}"
        f"{'PhMean(°)':>{num_w}}"
        f"{'PhStd(°)':>{num_w}}"
        f"{'MaxAbsErr':>{num_w}}"
        f"{'Status':>{9}}"
    )
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * (col_w + num_w * 7 + 9))

    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        row = (
            f"{r['model']:<{col_w}}"
            f"{r['n_samples']:>{num_w}}"
            f"{r['n_failed']:>{num_w}}"
            f"{r['amp_mean']:>{num_w}.6f}"
            f"{r['amp_std']:>{num_w}.2e}"
            f"{r['phase_mean']:>{num_w}.4f}"
            f"{r['phase_std']:>{num_w}.4f}"
            f"{r.get('max_abs_err', float('nan')):>{num_w}.2e}"
            f"{status:>{9}}"
        )
        terminalreporter.write_line(row)

    terminalreporter.write_line("-" * (col_w + num_w * 7 + 9))
    overall = "ALL PASSED" if all_passed else "SOME FAILED"
    terminalreporter.write_line(f"Overall: {overall}")
    terminalreporter.write_sep("=", "")

    # Persist metadata
    def _run_tag(r):
        return f"n{r['n_samples']}"

    sorted_results = sorted(results, key=_run_tag)
    for tag, group in groupby(sorted_results, key=_run_tag):
        group_list = list(group)
        results_dir = Path(__file__).parent / "results" / tag
        results_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"hardware": hw, "models": group_list}
        metadata_file = results_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        terminalreporter.write_line(f"Metadata saved to: {metadata_file}")
