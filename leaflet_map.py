import streamlit.components.v1 as components
from pathlib import Path

_HERE = Path(__file__).parent
_component_func = None


def _get_component():
    global _component_func
    if _component_func is None:
        _component_func = components.declare_component(
            "gpx_map",
            path=str(_HERE / "frontend"),
        )
    return _component_func


def render_map(data: dict, height: int = 520, key: str = "gpx_map"):
    payload = {
        "trkpts": [list(p) for p in data.get("trkpts", [])],
        "acpts":  data.get("acpts", []),
        "wpts":   data.get("wpts", []),
        "center": data.get("center", {"lat": 35.681, "lng": 139.767}),
        "zoom":   int(data.get("zoom", 13)),
        "force_center":       bool(data.get("force_center", False)),
        "click_threshold_px": int(data.get("click_threshold_px", 20)),
    }
    return _get_component()(data=payload, height=height, key=key, default=None)
