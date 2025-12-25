from __future__ import annotations

import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from flask import Blueprint, current_app, jsonify, request, send_file, send_from_directory
from fmpy import platform as fmpy_platform
from fmpy import simulate_fmu

from .config import AppConfig
from .fmu_service import generate_template, make_fmi_call_logger, platform_supports_fmu, validate_if_enabled
from .inputs import build_structured_input
from .storage import SessionStore, cleanup_old_files


api = Blueprint("api", __name__)


def _get_config() -> AppConfig:
    return current_app.config["APP_CONFIG"]


def _get_store() -> SessionStore:
    return current_app.extensions["session_store"]


def _ensure_upload_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@api.route("/")
def index():
    cfg = _get_config()
    return send_from_directory(str(cfg.static_dir), "index.html")


@api.post("/api/clear-session")
def clear_session():
    try:
        cfg = _get_config()
        result = _get_store().clear(cfg.upload_dir)
        return jsonify(
            {
                "ok": True,
                "message": f"Cleared {len(result['removed'])} file(s)",
                "removed": result["removed"],
                "errors": result["errors"],
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 500


@api.post("/api/upload-fmu")
def upload_fmu():
    cfg = _get_config()
    store = _get_store()
    try:
        print("Debug: Starting file upload")

        clear_first = request.form.get("clearFirst", "false").lower() == "true"
        if clear_first:
            print("Debug: Clearing previous session before upload")
            store.clear(cfg.upload_dir)

        if "file" not in request.files:
            print("Debug: No file part in request")
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]
        print(f"Debug: Received file: {file.filename}")

        if not file or file.filename == "":
            print("Debug: No file selected")
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".fmu"):
            print("Debug: Invalid file type")
            return jsonify({"error": "Only .fmu files allowed"}), 400

        print(f"Debug: Creating/verifying upload directory: {cfg.upload_dir}")
        _ensure_upload_dir(cfg.upload_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        fmu_path = cfg.upload_dir / safe_filename
        print(f"Debug: Saving to {fmu_path}")

        file.save(fmu_path)
        print("Debug: File saved successfully")

        print("Debug: Generating template...")
        tmpl = generate_template(str(fmu_path))
        print("Debug: Template generated successfully")

        store.set_fmu_path(str(fmu_path))
        print("Debug: Upload and processing completed successfully")
        return jsonify({"ok": True, "template": tmpl, "fmuPath": str(fmu_path)})

    except Exception as exc:
        print(f"Debug: Error in upload_fmu: {str(exc)}")
        print(f"Debug: Traceback: {traceback.format_exc()}")
        if "fmu_path" in locals() and Path(fmu_path).exists():
            try:
                Path(fmu_path).unlink()
                print(f"Debug: Removed partially uploaded file: {fmu_path}")
            except Exception as cleanup_error:
                print(f"Debug: Error during cleanup: {str(cleanup_error)}")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "trace": traceback.format_exc(),
                }
            ),
            500,
        )


@api.post("/api/run")
def run_simulation():
    print("Running simulation")
    payload = request.get_json(force=True)
    store = _get_store()
    fmu_path = payload.get("fmu") or store.fmu_path
    if not fmu_path or not os.path.exists(fmu_path):
        return jsonify({"ok": False, "error": "FMU not found. Upload first."}), 400

    logs: List[str] = []
    fmi_logger = make_fmi_call_logger(logs) if payload.get("debug_logging") else None

    raw_inputs = payload.get("inputs", None)
    signals = build_structured_input(raw_inputs)
    input_file = payload.get("input_file")

    kwargs = dict(
        filename=fmu_path,
        validate=payload.get("validate"),
        start_time=payload.get("start_time"),
        stop_time=payload.get("stop_time"),
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
        fmi_call_logger=fmi_logger,
    )
    if "remote_platform" in payload:
        kwargs["remote_platform"] = payload.get("remote_platform", "auto")

    if input_file:
        if not os.path.exists(input_file):
            return jsonify({"ok": False, "error": "Input file not found. Upload again."}), 400
        kwargs["input_file"] = input_file
    elif signals is not None:
        kwargs["input"] = signals

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        do_validate = payload.get("validate", True)
        if do_validate:
            problems = validate_if_enabled(fmu_path, do_validate)
            if problems:
                for problem in problems:
                    logs.append(f"{problem}")

        remote_platform = payload.get("remote_platform", "auto")
        can_sim, platform_logs = platform_supports_fmu(fmu_path, remote_platform)
        if not can_sim:
            logs.extend(platform_logs)
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"FMU does not support the current platform ({fmpy_platform}).",
                        "logs": logs,
                    }
                ),
                400,
            )

        result = simulate_fmu(**kwargs)

        df = pd.DataFrame(result)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = Path(fmu_path).parent / f"result_{ts}.csv"
        df.to_csv(out_csv, index=False)

        store.add_result_file(str(out_csv))

        return jsonify(
            {
                "ok": True,
                "columns": df.columns.tolist(),
                "rows": df.to_dict(orient="records"),
                "csv": str(out_csv),
                "logs": logs,
                "total_rows": len(df),
            }
        )

    except Exception as exc:
        msg = str(exc)
        if any("TwinCAT" in line for line in logs) or "TwinCAT" in msg:
            logs.append(
                "Hint: Install TwinCAT 3 XAE + XAR (matching build) so registry keys like 'DataDir' exist."
            )
        return (
            jsonify({"ok": False, "error": msg, "logs": logs, "trace": traceback.format_exc()}),
            500,
        )


@api.post("/api/upload-input")
def upload_input_file():
    cfg = _get_config()
    store = _get_store()
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No file"}), 400
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"ok": False, "error": "No file selected"}), 400
        if not file.filename.lower().endswith(".csv"):
            return jsonify({"ok": False, "error": "Only .csv files allowed"}), 400

        _ensure_upload_dir(cfg.upload_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        input_path = cfg.upload_dir / safe_filename
        file.save(input_path)

        store.set_input_file(str(input_path))
        return jsonify({"ok": True, "path": str(input_path)})
    except Exception as exc:
        return (
            jsonify({"ok": False, "error": str(exc), "type": type(exc).__name__, "trace": traceback.format_exc()}),
            500,
        )


@api.get("/api/download")
def download():
    cfg = _get_config()
    store = _get_store()
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "File not found"}), 404

    allowed = set(store.result_files)
    if store.fmu_path:
        allowed.add(store.fmu_path)
    if store.input_file:
        allowed.add(store.input_file)

    resolved = str(Path(path).resolve())
    if resolved not in allowed or not Path(resolved).exists():
        return jsonify({"error": "File not found"}), 404

    if not str(resolved).startswith(str(cfg.upload_dir.resolve())):
        return jsonify({"error": "File not found"}), 404

    try:
        return send_file(resolved, as_attachment=True, download_name=os.path.basename(resolved))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@api.get("/<path:path>")
def statics(path):
    cfg = _get_config()
    return send_from_directory(str(cfg.static_dir), path)


def startup_cleanup() -> None:
    cfg = _get_config()
    removed = cleanup_old_files(cfg.upload_dir, cfg.max_upload_age_hours)
    if removed:
        print(f"Cleanup complete: removed {removed} old file(s)")
