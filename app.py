
#!/usr/bin/env python3
"""
Generalized FMU runner (FMPy) with config generation and execution.

Features:
- --generate-template <FMU> : creates a JSON config listing parameters/inputs/outputs
- --run <config.json>       : runs simulate_fmu() using the config
- Optional plotting and validation

Supports FMI 1.0, 2.0, and 3.0 FMUs via FMPy's simulate_fmu().
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List

import pandas as pd
from fmpy import read_model_description, simulate_fmu
from fmpy.util import plot_result
from fmpy.validation import validate_fmu


# -----------------------------
# Helpers
# -----------------------------

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def list_variables(fmu_path: str) -> List[Dict[str, Any]]:
    """Return variable info (name, type, causality, variability, start) from FMU."""
    md = read_model_description(fmu_path)
    vars_info = []
    for v in md.modelVariables:
        vars_info.append({
            "name": v.name,
            "type": v.type,
            "causality": v.causality,
            "variability": getattr(v, "variability", None),
            "start": getattr(v, "start", None),
            "description": getattr(v, "description", None)
        })
    return vars_info


def generate_template(fmu_path: str,
                      out_path: str = "config.json",
                      include_input_skeleton: bool = True) -> Dict[str, Any]:
    """
    Generate a config template based on FMU metadata:
    - parameters -> start_values with their default starts
    - inputs     -> empty schedules (or single [0, start] if available)
    - outputs    -> list of output variable names
    - time settings -> from defaultExperiment if present
    """
    md = read_model_description(fmu_path)

    # Default experiment times/tolerance (if provided)
    defexp = getattr(md, "defaultExperiment", None)
    start_time = getattr(defexp, "startTime", None)
    stop_time = getattr(defexp, "stopTime", None)
    rel_tol = getattr(defexp, "tolerance", None)

    # Reasonable fallbacks
    if start_time is None:
        start_time = 0.0
    if stop_time is None:
        stop_time = start_time + 10.0

    # Gather variables by causality
    params = []
    inputs = []
    outputs = []

    for v in md.modelVariables:
        if v.causality == "parameter":
            params.append(v)
        elif v.causality == "input":
            inputs.append(v)
        elif v.causality == "output":
            outputs.append(v)

    # Build start_values from parameters (use their start if provided)
    start_values: Dict[str, Any] = {}
    for p in params:
        sv = getattr(p, "start", None)
        # Keep None if not present; user will fill
        start_values[p.name] = sv

    # Input schedules skeleton
    input_schedules = []
    for inp in inputs:
        if include_input_skeleton:
            # If input has a start value we can seed it at t=0, else leave empty
            sv = getattr(inp, "start", None)
            if sv is not None:
                input_schedules.append([inp.name, [[0.0, sv]]])
            else:
                input_schedules.append([inp.name, []])
        else:
            input_schedules.append([inp.name, []])

    # Output names
    output_names = [o.name for o in outputs]

    # Determine FMI types provided (ME/CS)
    provides_me = md.modelExchange is not None
    provides_cs = md.coSimulation is not None

    # fmi_type: let user force, otherwise auto by leaving null
    fmi_type = None
    solver = "CVode" if provides_me else None

    cfg = {
        "fmu": os.path.abspath(fmu_path),

        # Time settings
        "start_time": start_time,
        "stop_time": stop_time,

        # Sampling cadence (choose one, or omit both for FMPy defaults)
        "step_size": 0.01,           # good starting point; user can adjust
        # "output_interval": 0.01,

        # Numerical options
        "solver": solver,                         # for ME; ignored for CS
        "relative_tolerance": rel_tol if rel_tol is not None else 1e-5,
        "record_events": True,                    # ME only
        "fmi_type": fmi_type,                     # None = auto

        # Model configuration
        "start_values": start_values,             # parameters (and could be more)
        "inputs": input_schedules,                # schedules per input variable
        "outputs": output_names,                  # default: record outputs

        # Diagnostics & UX
        "validate": True,
        "timeout": 60,
        "debug_logging": False,
        "visible": False,
        "set_stop_time": True,

        # Results
        "output_csv": "result.csv",
        "plot_png": "result.png"
    }

    # Persist to file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Template written to {out_path}")
    return cfg


import numpy as np

def normalize_inputs(input_cfg):
    """Convert any string numbers to floats and keep the same [name, [[t, v],...]] shape."""
    if not input_cfg:
        return None
    normalized = []
    for name, pairs in input_cfg:
        fixed_pairs = []
        for t, v in pairs:
            fixed_pairs.append([float(t), float(v)])
        normalized.append([name, fixed_pairs])
    return normalized

def build_structured_input(input_cfg):
    """
    Convert config format:
        [ [name, [[t, v], [t, v], ...]], ... ]
    → structured NumPy array with dtype [('time', float64), (name1, float64), ...]
    using zero-order hold between time points.
    Returns None if there are no samples.
    """
    if not input_cfg:  # None or []
        return None

    # 1) normalize (strings -> floats)
    input_cfg = normalize_inputs(input_cfg)

    # 2) collect union of time stamps
    series = {}
    all_times = set()
    for name, samples in input_cfg:
        ts = [(float(t), float(v)) for t, v in samples]
        ts.sort(key=lambda x: x[0])
        series[name] = ts
        for t, _ in ts:
            all_times.add(t)

    if not all_times:
        return None  # nothing to set

    times = np.array(sorted(all_times), dtype=np.float64)

    # 3) build structured array
    dtype = [('time', np.float64)] + [(n, np.float64) for n in series.keys()]
    data = np.zeros(times.shape[0], dtype=dtype)
    data['time'] = times

    # 4) fill each signal with zero-order hold
    for n, ts in series.items():
        if not ts:
            data[n] = 0.0
            continue
        idx = 0
        last = ts[0][1]
        for i, t in enumerate(times):
            while idx + 1 < len(ts) and ts[idx + 1][0] <= t + 1e-15:
                idx += 1
                last = ts[idx][1]
            data[n][i] = last

    return data



def run_from_config(cfg_path: str) -> pd.DataFrame:
    """Load config JSON and execute simulate_fmu()."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fmu_path = cfg.get("fmu")
    if not fmu_path or not os.path.exists(fmu_path):
        raise FileNotFoundError(f"FMU not found: {fmu_path}")

    # Optional validation (prints messages only)
    if cfg.get("validate", True):
        problems = validate_fmu(fmu_path)
        if problems:
            print("Validation messages:")
            for p in problems:
                print(f"  [{p.severity}] {p.message}")

    
    raw_inputs = cfg.get("inputs", None)
    signals = build_structured_input(raw_inputs)

    kwargs = dict(
        filename=fmu_path,
        start_time=cfg.get("start_time"),
        stop_time=cfg.get("stop_time"),
        # For Co-Simulation prefer uniform logging:
        output_interval=cfg.get("output_interval"),
        # ME-only options (ignored for CS):
        solver=cfg.get("solver"),
        relative_tolerance=cfg.get("relative_tolerance"),
        record_events=cfg.get("record_events"),
        fmi_type=cfg.get("fmi_type"),
        start_values=cfg.get("start_values"),
        output=cfg.get("outputs"),
        timeout=cfg.get("timeout"),
        debug_logging=cfg.get("debug_logging"),
        visible=cfg.get("visible"),
        set_stop_time=cfg.get("set_stop_time", True),
        # optional: fmi_call_logger=fmi_call_logger,
    )

    # ✅ Only include 'input' when we actually built a structured array
    if signals is not None:
        kwargs["input"] = signals

    # Drop None-valued keys so FMPy uses defaults
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    result = simulate_fmu(**kwargs)


    # Save CSV
    out_csv = cfg.get("output_csv", "result.csv")
    df = pd.DataFrame(result)
    df.to_csv(out_csv, index=False)
    print(f"Simulation complete. Results saved to {out_csv}")

    # Optional plot
    plot_path = cfg.get("plot_png")
    if plot_path:
        try:
            plot_result(result)
            import matplotlib.pyplot as plt
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")

    return df


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generalized FMU runner (FMPy) with config generation."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # generate-template
    p_gen = sub.add_parser("generate-template", help="Generate a config JSON from FMU metadata.")
    p_gen.add_argument("fmu", help="Path to FMU file")
    p_gen.add_argument("-o", "--out", default="config.json", help="Output JSON path")
    p_gen.add_argument("--no-input-skeleton", action="store_true",
                       help="Do not pre-fill input schedules with start values")

    # list-variables
    p_ls = sub.add_parser("list-variables", help="List variables (name, type, causality, start).")
    p_ls.add_argument("fmu", help="Path to FMU file")

    # run
    p_run = sub.add_parser("run", help="Run simulation from a config JSON.")
    p_run.add_argument("config", help="Path to config JSON")

    args = p.parse_args()

    if args.cmd == "generate-template":
        generate_template(args.fmu, args.out, include_input_skeleton=not args.no_input_skeleton)

    elif args.cmd == "list-variables":
        vars_info = list_variables(args.fmu)
        print("Variables:")
        for v in vars_info:
            print(f"  {v['name']:30}  {v['type']:7}  {v['causality']:12}  start={v['start']}")

    elif args.cmd == "run":
        run_from_config(args.config)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
