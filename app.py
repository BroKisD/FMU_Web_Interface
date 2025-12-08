
#!/usr/bin/env python3
import os
import io
import json
import tempfile
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory

from fmpy import read_model_description, simulate_fmu
from fmpy import platform as fmpy_platform
from fmpy.validation import validate_fmu


app = Flask(__name__, static_url_path='', static_folder='static')

# Simple in-memory store for last uploaded FMU per process (for demo)
SESSION: Dict[str, Any] = {}


# ---------- Helpers ----------

def normalize_inputs(input_cfg: Optional[List]) -> Optional[List]:
    """Convert string numbers to floats; keep [name, [[t, v], ...]] shape."""
    if not input_cfg:
        return None
    normalized = []
    for name, pairs in input_cfg:
        fixed_pairs = []
        for t, v in pairs:
            fixed_pairs.append([float(t), float(v)])
        normalized.append([name, fixed_pairs])
    return normalized


def build_structured_input(input_cfg: Optional[List]) -> Optional[np.ndarray]:
    """
    Convert config format [ [name, [[t, v], ...]], ... ] to a structured NumPy array
    dtype: [('time', float64), (name1, float64), ...] with ZOH between points.
    Returns None if no samples.
    """
    if not input_cfg:
        return None

    input_cfg = normalize_inputs(input_cfg)

    series = {}
    all_times = set()
    for name, samples in input_cfg:
        ts = [(float(t), float(v)) for t, v in samples]
        ts.sort(key=lambda x: x[0])
        series[name] = ts
        for t, _ in ts:
            all_times.add(t)

    if not all_times:
        return None

    times = np.array(sorted(all_times), dtype=np.float64)
    dtype = [('time', np.float64)] + [(n, np.float64) for n in series.keys()]
    data = np.zeros(times.shape[0], dtype=dtype)
    data['time'] = times

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


def summarize_variables(md) -> Dict[str, Any]:
    """Return variables grouped by causality + defaults."""
    params, inputs, outputs, indep = [], [], [], []
    seen = set()
    for v in md.modelVariables:
        if v.name in seen:
            continue
        seen.add(v.name)
        entry = {
            "name": v.name,
            "type": v.type,
            "causality": v.causality,
            "variability": getattr(v, "variability", None),
            "start": getattr(v, "start", None),
            "description": getattr(v, "description", None),
        }
        if v.causality == "parameter":
            params.append(entry)
        elif v.causality == "input":
            inputs.append(entry)
        elif v.causality == "output":
            outputs.append(entry)
        elif v.causality == "independent":
            indep.append(entry)
    return {"parameters": params, "inputs": inputs, "outputs": outputs, "independent": indep}


def generate_template(fmu_path: str) -> Dict[str, Any]:
    """Create a config template using FMU metadata (general for any FMU)."""
    md = read_model_description(fmu_path)
    defexp = getattr(md, "defaultExperiment", None)
    start_time = getattr(defexp, "startTime", None) or 0.0
    stop_time = getattr(defexp, "stopTime", None) or (start_time + 10.0)
    rel_tol = getattr(defexp, "tolerance", None) or 1e-5

    groups = summarize_variables(md)

    # Build start_values from parameters
    start_values = {p["name"]: p["start"] for p in groups["parameters"]}

    # Input schedules skeleton (seed t=0 with start if present)
    input_schedules = []
    for inp in groups["inputs"]:
        sv = inp["start"]
        schedule = [[0.0, sv]] if (sv is not None) else []
        input_schedules.append([inp["name"], schedule])

    # Suggested outputs = all outputs (user can edit)
    output_names = [o["name"] for o in groups["outputs"]]

    # Prefer Co-Simulation if provided; solver is for ME only
    cfg = {
        "fmu": fmu_path,
        "fmi_type": None,  # auto (user can force "CoSimulation" or "ModelExchange")
        "start_time": float(start_time),
        "stop_time": float(stop_time),

        # Prefer uniform logging via output_interval; user can switch/remove
        "output_interval": 0.01,

        # ME options (ignored for CS)
        "solver": "CVode" if md.modelExchange is not None else None,
        "relative_tolerance": float(rel_tol),
        "record_events": True,

        # Model configuration
        "start_values": start_values,
        "inputs": input_schedules if input_schedules else None,
        "outputs": output_names if output_names else None,

        # UX & artifacts
        "validate": True,
        "timeout": 60,
        "debug_logging": False,
        "visible": False,
        "set_stop_time": True,

        "output_csv": "result.csv",
        "plot_png": "result.png"
    }
    return {"config": cfg, "variables": groups, "fmiVersion": md.fmiVersion,
            "provides": {"coSimulation": md.coSimulation is not None,
                         "modelExchange": md.modelExchange is not None},
            "platform": fmpy_platform(),
            "info": fmu_info(fmu_path)}


def make_fmi_call_logger(buffer: List[str]):
    """Capture FMI calls/log lines in-memory for returning to the UI."""
    def logger(*args):
        try:
            if len(args) == 1:
                buffer.append(str(args[0]))
            else:
                # comp, name, status, message (format varies)
                comp, name, status, message = args
                buffer.append(f"[FMI] {name} -> {status} | {message}")
        except Exception:
            buffer.append(" ".join(str(a) for a in args))
    return logger


# ---------- Routes ----------

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.post('/api/upload-fmu')
def upload_fmu():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.fmu'):
        return jsonify({"error": "Only .fmu files allowed"}), 400

    td = tempfile.mkdtemp(prefix="fmu_")
    fmu_path = os.path.join(td, file.filename)
    file.save(fmu_path)

    try:
        tmpl = generate_template(fmu_path)
        # Remember last FMU
        SESSION['fmu_path'] = fmu_path
        return jsonify({"ok": True, "template": tmpl, "fmuPath": fmu_path})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post('/api/run')
def run_simulation():
    payload = request.get_json(force=True)
    fmu_path = payload.get("fmu") or SESSION.get("fmu_path")
    if not fmu_path or not os.path.exists(fmu_path):
        return jsonify({"ok": False, "error": "FMU not found. Upload first."}), 400

    # Build kwargs for simulate_fmu
    logs: List[str] = []
    fmi_logger = make_fmi_call_logger(logs) if payload.get("debug_logging") else None

    # Prepare inputs
    raw_inputs = payload.get("inputs", None)
    signals = build_structured_input(raw_inputs)

    kwargs = dict(
        filename=fmu_path,
        start_time=payload.get("start_time"),
        stop_time=payload.get("stop_time"),
        # Prefer output_interval for CS; ME-only fields are accepted but may be ignored:
        output_interval=payload.get("output_interval"),
        step_size=payload.get("step_size"),
        solver=payload.get("solver"),
        relative_tolerance=payload.get("relative_tolerance"),
        record_events=payload.get("record_events"),
        fmi_type=payload.get("fmi_type"),
        start_values=payload.get("start_values"),
        output=payload.get("outputs"),
        timeout=payload.get("timeout"),
        debug_logging=payload.get("debug_logging"),
        visible=payload.get("visible"),
        set_stop_time=payload.get("set_stop_time", True),
        fmi_call_logger=fmi_logger
    )
    # Only add input if we have signals
    if signals is not None:
        kwargs["input"] = signals

    # Drop None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        # Optional validation
        if payload.get("validate", True):
            problems = validate_fmu(fmu_path)
            if problems:
                for p in problems:
                    logs.append(f"[VALIDATION] [{p.severity}] {p.message}")

        result = simulate_fmu(**kwargs)

        # DataFrame for CSV and slicing
        df = pd.DataFrame(result)
        # Save CSV to temp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(os.path.dirname(fmu_path), f"result_{ts}.csv")
        df.to_csv(out_csv, index=False)

        # Optional PNG plot
        out_png = None
        if payload.get("plot_png"):
            try:
                from matplotlib import pyplot as plt
                plot_result(result)
                out_png = os.path.join(os.path.dirname(fmu_path), f"plot_{ts}.png")
                plt.savefig(out_png, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logs.append(f"[PLOT] Failed: {e}")

        # Return a small preview to the UI and download links
        preview_rows = min(len(df), 500)
        return jsonify({
            "ok": True,
            "columns": df.columns.tolist(),
            "rows": df.head(preview_rows).to_dict(orient="records"),
            "csv": out_csv,
            "png": out_png,
            "logs": logs
        })
    except Exception as e:
        # Surface common TwinCAT dependency hint if present
        msg = str(e)
        if any("TwinCAT" in line for line in logs) or "TwinCAT" in msg:
            logs.append("Hint: Install TwinCAT 3 XAE + XAR (matching build) so registry keys like 'DataDir' exist.")
        return jsonify({"ok": False, "error": msg, "logs": logs, "trace": traceback.format_exc()}), 500


@app.get('/api/download')
def download():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    # Secure: restrict to temp dirs created in this process
    try:
        return send_file(path, as_attachment=True, download_name=os.path.basename(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve the static files (index.html, etc.)
@app.get('/<path:path>')
def statics(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    # For local dev; use a proper WSGI server in production (gunicorn / waitress)
    app.run(host='127.0.0.1', port=8000, debug=True)
