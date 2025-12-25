from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fmpy import read_model_description, supported_platforms
from fmpy import platform as fmpy_platform
from fmpy.util import can_simulate
from fmpy.validation import validate_fmu


def get_supported_platforms(fmu_path: str) -> List[str]:
    try:
        return supported_platforms(fmu_path)
    except Exception as exc:
        print(f"Debug: Failed to read supported platforms: {exc}")
        return []


def summarize_variables(md) -> Dict[str, Any]:
    params, inputs, outputs, indep = [], [], [], []
    seen = set()
    for var in md.modelVariables:
        if var.name in seen:
            continue
        seen.add(var.name)
        entry = {
            "name": var.name,
            "type": var.type,
            "causality": var.causality,
            "variability": getattr(var, "variability", None),
            "start": getattr(var, "start", None),
            "description": getattr(var, "description", None),
        }
        if var.causality == "parameter":
            params.append(entry)
        elif var.causality == "input":
            inputs.append(entry)
        elif var.causality == "output":
            outputs.append(entry)
        elif var.causality == "independent":
            indep.append(entry)
    return {"parameters": params, "inputs": inputs, "outputs": outputs, "independent": indep}


def generate_template(fmu_path: str) -> Dict[str, Any]:
    md = read_model_description(fmu_path)
    defexp = getattr(md, "defaultExperiment", None)
    start_time = getattr(defexp, "startTime", None) or 0.0
    stop_time = getattr(defexp, "stopTime", None) or (start_time + 10.0)
    rel_tol = getattr(defexp, "tolerance", None) or 1e-5

    groups = summarize_variables(md)

    start_values = {p["name"]: p["start"] for p in groups["parameters"]}

    input_schedules = []
    for inp in groups["inputs"]:
        sv = inp["start"]
        schedule = [[0.0, sv]] if (sv is not None) else []
        input_schedules.append([inp["name"], schedule])

    output_names = [o["name"] for o in groups["outputs"]]
    platforms = get_supported_platforms(fmu_path)

    cfg = {
        "fmu": fmu_path,
        "fmi_type": None,
        "start_time": float(start_time),
        "stop_time": float(stop_time),
        "output_interval": 0.01,
        "solver": "CVode" if md.modelExchange is not None else None,
        "relative_tolerance": float(rel_tol),
        "record_events": True,
        "start_values": start_values,
        "inputs": input_schedules if input_schedules else None,
        "outputs": output_names if output_names else None,
        "validate": True,
        "timeout": 60,
        "debug_logging": True,
        "visible": False,
        "set_stop_time": True,
        "output_csv": "result.csv",
        "input_file": None,
    }
    return {
        "config": cfg,
        "variables": groups,
        "fmiVersion": md.fmiVersion,
        "provides": {
            "coSimulation": md.coSimulation is not None,
            "modelExchange": md.modelExchange is not None,
        },
        "platform": fmpy_platform,
        "supported_platforms": platforms,
        "info": {
            "fmiVersion": md.fmiVersion,
            "modelName": getattr(md, "modelName", "Unknown"),
            "description": getattr(md, "description", "No description available"),
            "generationTool": getattr(md, "generationTool", "Unknown"),
            "generationDateAndTime": getattr(md, "generationDateAndTime", "Unknown"),
            "supportedPlatforms": ", ".join(platforms) if platforms else "unknown",
        },
    }


def validate_if_enabled(fmu_path: str, enabled: bool) -> List[str]:
    if not enabled:
        return []
    return validate_fmu(fmu_path)


def platform_supports_fmu(fmu_path: str, remote_platform: Optional[str]) -> Tuple[bool, List[str]]:
    platforms = get_supported_platforms(fmu_path)
    can_sim, _ = can_simulate(platforms, remote_platform)
    logs = []
    if not can_sim:
        platforms_display = ", ".join(platforms) if platforms else "none"
        logs.append(f"FMU platforms: {platforms_display}")
        logs.append(f"Host platform: {fmpy_platform}")
    return can_sim, logs


def make_fmi_call_logger(buffer: List[str]):
    def logger(*args):
        try:
            if len(args) == 1:
                buffer.append(str(args[0]))
            else:
                comp, name, status, message = args
                buffer.append(f"[FMI] {name} -> {status} | {message}")
        except Exception:
            buffer.append(" ".join(str(a) for a in args))
    return logger
