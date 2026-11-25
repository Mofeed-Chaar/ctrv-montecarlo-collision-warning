import numpy as np
import cv2


def add_probability_colorbar(
    img,
    bar_width=40,
    pad=20,
    ticks=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
    title="p/p_max",
    start_frac=0.5,   # يبدأ من منتصف الصورة (0.5)
    height_frac=0.5   # طوله نصف الصورة (0.5) -> من المنتصف إلى الأسفل
):
    """
    Adds a vertical colorbar on the right side:
    red = high probability (1), green = low probability (0).

    start_frac: where the bar starts vertically (0..1) of image height.
    height_frac: bar height as fraction (0..1) of image height.
    """
    H, W = img.shape[:2]
    out = img.copy()

    # Compute bar placement/size
    y0 = int(round(H * start_frac)) + pad
    bar_h = int(round(H * height_frac)) - 2 * pad

    # Clamp to image bounds
    y0 = max(0, min(H - 1, y0))
    bar_h = max(0, min(H - y0, bar_h))

    if bar_h <= 10:
        return out

    # pnorm from top->bottom: 1 -> 0
    pvals = np.linspace(1.0, 0.0, bar_h).astype(np.float32)

    # Build BGR colors: R=255*p, G=255*(1-p)
    bar = np.zeros((bar_h, bar_width, 3), dtype=np.uint8)
    bar[..., 2] = (255.0 * pvals[:, None]).astype(np.uint8)          # R
    bar[..., 1] = (255.0 * (1.0 - pvals[:, None])).astype(np.uint8)  # G
    bar[..., 0] = 0                                                  # B

    # Place bar on right side
    x0 = W - pad - bar_width
    if x0 < 0:
        return out

    out[y0:y0 + bar_h, x0:x0 + bar_width] = bar

    # Outline
    cv2.rectangle(out, (x0, y0), (x0 + bar_width, y0 + bar_h), (0, 0, 0), 1)

    # Title (put it above the bar)
    title_y = max(15, y0 - 8)
    cv2.putText(out, title, (max(5, x0 - 10), title_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Ticks and labels (relative to bar height)
    for t in ticks:
        t = float(np.clip(t, 0.0, 1.0))
        yy = int(round(y0 + (1.0 - t) * (bar_h - 1)))
        cv2.line(out, (x0 - 6, yy), (x0, yy), (0, 0, 0), 1)
        cv2.putText(out, f"{t:.1f}", (x0 - 45, yy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return out

def draw_gaussian_heatmap_world_on_image(
    img,
    mu_world,
    Sigma_world,
    s=16.0,
    offset_px=(0, 0),
    flip_y=False,
    confidence=0.99,
    alpha=0.6,
    draw_ellipse=True,
    ellipse_color=(0, 0, 255),
    ellipse_thickness=2,
    add_colorbar=True
):
    img = img.copy()
    mu_world = np.array(mu_world, dtype=float).reshape(2)
    Sigma_world = np.array(Sigma_world, dtype=float).reshape(2, 2)

    # world -> px
    mu_px = s * mu_world
    Sigma_px = (s**2) * Sigma_world

    if flip_y:
        mu_px[1] = -mu_px[1]
        J = np.array([[1, 0],
                      [0, -1]], dtype=float)
        Sigma_px = J @ Sigma_px @ J.T

    ox, oy = offset_px
    cx = float(mu_px[0] + ox)
    cy = float(mu_px[1] + oy)

    chi2_map = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
    if confidence not in chi2_map:
        raise ValueError(f"confidence must be one of {list(chi2_map.keys())}")
    chi2 = chi2_map[confidence]

    # ellipse size
    eigvals, eigvecs = np.linalg.eigh(Sigma_px)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    a = np.sqrt(eigvals[0] * chi2)
    b = np.sqrt(eigvals[1] * chi2)

    # computation ROI: axis-aligned bounding box of the (rotated) ellipse.
    # Half-extent along x is sqrt(chi2 * Sxx), along y sqrt(chi2 * Syy).
    # (Using a / b here clipped every ellipse that is not aligned with the x axis.)
    hx = np.sqrt(chi2 * Sigma_px[0, 0])
    hy = np.sqrt(chi2 * Sigma_px[1, 1])
    margin = 6
    x0 = int(np.floor(cx - hx - margin))
    x1 = int(np.ceil (cx + hx + margin))
    y0 = int(np.floor(cy - hy - margin))
    y1 = int(np.ceil (cy + hy + margin))

    H, W = img.shape[:2]
    x0c, x1c = max(0, x0), min(W - 1, x1)
    y0c, y1c = max(0, y0), min(H - 1, y1)
    if x0c >= x1c or y0c >= y1c:
        return add_probability_colorbar(img) if add_colorbar else img

    xs = np.arange(x0c, x1c + 1)
    ys = np.arange(y0c, y1c + 1)
    X, Y = np.meshgrid(xs, ys)

    dx = X - cx
    dy = Y - cy

    Sigma_inv = np.linalg.inv(Sigma_px)
    d2 = (Sigma_inv[0, 0] * dx * dx +
          (Sigma_inv[0, 1] + Sigma_inv[1, 0]) * dx * dy +
          Sigma_inv[1, 1] * dy * dy)

    mask = (d2 <= chi2)

    # TRUE Gaussian relative probability: exp(-d2/2) is already 1 at the mean.
    # (Dividing by the max of the visible part made the colours wrong when the
    # mean lies outside the image.)
    pnorm = np.zeros_like(d2, dtype=np.float32)
    pnorm[mask] = np.exp(-0.5 * d2[mask])

    # color: red high, green low
    heat = np.zeros((pnorm.shape[0], pnorm.shape[1], 3), dtype=np.uint8)
    heat[..., 2] = (255.0 * pnorm).astype(np.uint8)
    heat[..., 1] = (255.0 * (1.0 - pnorm)).astype(np.uint8)

    roi = img[y0c:y1c+1, x0c:x1c+1]
    out_roi = roi.copy()
    m = mask.astype(bool)
    out_roi[m] = (alpha * heat[m] + (1.0 - alpha) * roi[m]).astype(np.uint8)
    img[y0c:y1c+1, x0c:x1c+1] = out_roi

    # optional ellipse border
    if draw_ellipse:
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        center = (int(round(cx)), int(round(cy)))
        axes = (int(round(a)), int(round(b)))
        cv2.ellipse(img, center, axes, angle, 0, 360, ellipse_color, ellipse_thickness)
        cv2.circle(img, center, 3, ellipse_color, -1)

    if add_colorbar:
        img = add_probability_colorbar(img, title="p/p_max")

    return img


if __name__ == "__main__":
    img = np.ones((1600, 1600, 3), dtype=np.uint8) * 255

    mu_world = [31.6071918426, 12.1273783936]
    Sigma_world = [[0.40, 0.05],
                   [0.05, 0.30]]

    out = draw_gaussian_heatmap_world_on_image(
        img,
        mu_world=mu_world,
        Sigma_world=Sigma_world,
        s=16.0,
        offset_px=(800, 800),
        flip_y=False,
        confidence=0.99,
        alpha=0.60,
        draw_ellipse=True,
        add_colorbar=True
    )

    cv2.imwrite("gaussian_true_probability_with_colorbar.png", out)
    print("Saved: gaussian_true_probability_with_colorbar.png")