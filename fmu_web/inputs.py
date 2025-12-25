from __future__ import annotations

from typing import List, Optional

import numpy as np


def normalize_inputs(input_cfg: Optional[List]) -> Optional[List]:
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
    dtype = [("time", np.float64)] + [(n, np.float64) for n in series.keys()]
    data = np.zeros(times.shape[0], dtype=dtype)
    data["time"] = times

    for name, ts in series.items():
        if not ts:
            data[name] = 0.0
            continue
        idx = 0
        last = ts[0][1]
        for i, t in enumerate(times):
            while idx + 1 < len(ts) and ts[idx + 1][0] <= t + 1e-15:
                idx += 1
                last = ts[idx][1]
            data[name][i] = last

    return data
