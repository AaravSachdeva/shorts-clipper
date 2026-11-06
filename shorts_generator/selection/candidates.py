"""Stage 2: candidate clips for the deep selector, aiming for recall."""
import math
from typing import Dict, List, Tuple

from ..boundaries import snap_clip

SIGNALS_PER_HOUR = 20


def signal_spans(words: List[Dict], times: List[Tuple[float, str, float]], duration: float,
                 per_hour: int = SIGNALS_PER_HOUR) -> List[Dict]:
    """A 20-60 s sentence-snapped span around each (time, origin, strength); the strongest per_hour per hour."""
    cap = max(1, math.ceil(per_hour * duration / 3600))
    out = []
    for t, origin, strength in sorted(times, key=lambda x: x[2], reverse=True):
        if len(out) >= cap:
            break
        snapped = snap_clip(words, t - 20, t + 20, window=10.0)
        if snapped:
            out.append({**snapped, "origin": [origin], "strength": strength})
    return out
