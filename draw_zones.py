"""Draw the collision zones on the intersection map -> img/zones.png

Left : `zones`         (used by visualization.py)
Right: `collect_zones` (used by collect_data.py)

Run after changing the zones in config/parameters.py:
    python draw_zones.py
"""
import os

import cv2
import numpy as np
from config import parameters

param = parameters.ModelParameters()
data = parameters.DataConfig()

W, H = param.img_size                     # image size in pixels (1600 x 1600)
ppm = param.pixels_per_meter              # 16 px = 1 m
cx, cy = W // 2, H // 2                   # sensor origin in the image
out_path = parameters.project_path('img/zones.png')

ZONE_COLORS = [(0, 0, 230), (0, 150, 0), (230, 80, 0), (160, 0, 160), (0, 140, 200),
               (120, 120, 0), (0, 0, 120), (90, 60, 30)]


def to_metres(zone):
    """Pixel zone (x1, y1, x2, y2) -> sensor-frame ranges in metres.

    The image column u corresponds to the sensor y axis, the image row v to the
    sensor x axis: u = 800 + 16 * y,  v = 800 + 16 * x.
    """
    x1, y1, x2, y2 = zone
    return ((y1 - cy) / ppm, (y2 - cy) / ppm,      # sensor x range
            (x1 - cx) / ppm, (x2 - cx) / ppm)      # sensor y range


def draw_grid(img):
    """Metre grid every 10 m and the sensor origin."""
    for m in range(-50, 51, 10):
        p = int(cx + m * ppm)
        cv2.line(img, (p, 0), (p, H), (200, 200, 200), 1)
        cv2.line(img, (0, p), (W, p), (200, 200, 200), 1)
        if m != 0:
            cv2.putText(img, f'{m}', (p + 4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2, cv2.LINE_AA)
            cv2.putText(img, f'{m}', (4, p - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.drawMarker(img, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(img, 'origin (0, 0)', (cx + 8, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'sensor y [m] ->', (W - 300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(img, 'sensor x [m] (down)', (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA)


def draw_panel(zones, title):
    img = cv2.imread(data.background_image)
    if img is None:
        img = np.full((H, W, 3), 255, np.uint8)
    img = cv2.addWeighted(img, 0.55, np.full_like(img, 255), 0.45, 0)   # lighten the map
    draw_grid(img)

    for i, (x1, y1, x2, y2) in enumerate(zones):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        label = f'Zone {i + 1}'
        cv2.putText(img, label, (x1 + 8, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
        w_m, h_m = (x2 - x1) / ppm, (y2 - y1) / ppm
        cv2.putText(img, f'{w_m:.1f} x {h_m:.1f} m', (x1 + 8, y1 + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, H - 90), (W, H - 50), (255, 255, 255), -1)
    cv2.putText(img, title, (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (0, 0, 0), 2)
    return img


if __name__ == '__main__':
    left = draw_panel(param.zones, f'zones - visualization.py  ({len(param.zones)} zones)')
    right = draw_panel(param.collect_zones, f'collect_zones - collect_data.py  ({len(param.collect_zones)} zones)')
    gap = np.full((H, 30, 3), 255, np.uint8)
    out = np.hstack([left, gap, right])
    out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    print(f'Saved {out_path}')
    for name, zs in (('zones', param.zones), ('collect_zones', param.collect_zones)):
        print(name)
        for i, z in enumerate(zs):
            xa, xb, ya, yb = to_metres(z)
            print(f'  Zone {i + 1}: px {z}  ->  sensor x {xa:+.1f}..{xb:+.1f} m, y {ya:+.1f}..{yb:+.1f} m')
