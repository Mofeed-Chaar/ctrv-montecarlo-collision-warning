import numpy as np

def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi

def to_time_seconds(sec: int, nanosec: int) -> float:
    return float(sec) + float(nanosec) * 1e-9

def _robust_std(x):
    """Standard deviation after removing the 1% / 99% tails."""
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return float(np.std(x)) if len(x) else float('nan')
    lo, hi = np.quantile(x, [0.01, 0.99])
    x2 = x[(x >= lo) & (x <= hi)]
    return float(np.std(x2)) if len(x2) > 0 else float(np.std(x))


def _collect_derivatives(seq, min_lag=0.5):
    """Return (a, alpha) samples of one track.

    Derivatives are taken between samples at least `min_lag` seconds apart.
    With 10 Hz data, differencing consecutive frames (0.1 s) mostly measures
    sensor noise, not real acceleration.
    """
    seq = sorted(seq, key=lambda r: (r[0], r[1]))
    t = np.array([to_time_seconds(r[0], r[1]) for r in seq], dtype=float)
    v = np.array([r[4] for r in seq], dtype=float)
    omega = np.array([r[6] for r in seq], dtype=float)
    a, alpha = [], []
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < min_lag:
            j += 1
        if j >= len(t):
            break
        dt = t[j] - t[i]
        if dt > 3 * min_lag:          # gap in the track
            continue
        a.append((v[j] - v[i]) / dt)
        alpha.append((omega[j] - omega[i]) / dt)
    return a, alpha


def estimate_process_noise(objects, v_eps=1e-6, min_lag=0.5, by_class=False, min_samples=30):
    """Estimate sigma_a and sigma_alpha from the tracked history.

    by_class=False -> returns (sigma_a, sigma_alpha) over all objects (old behaviour).
    by_class=True  -> returns {obj_type: (sigma_a, sigma_alpha)}; classes with fewer
                      than `min_samples` samples are missing from the dict.
    """
    a_cls, alpha_cls = {}, {}
    for obj_id, seq in objects.items():
        if len(seq) < 3:
            continue
        a, alpha = _collect_derivatives(seq, min_lag=min_lag)
        obj_type = seq[-1][7]
        a_cls.setdefault(obj_type, []).extend(a)
        alpha_cls.setdefault(obj_type, []).extend(alpha)

    if by_class:
        out = {}
        for c in a_cls:
            if len(a_cls[c]) >= min_samples:
                out[c] = (_robust_std(a_cls[c]), _robust_std(alpha_cls[c]))
        return out

    a_all = [x for v in a_cls.values() for x in v]
    alpha_all = [x for v in alpha_cls.values() for x in v]
    if len(a_all) == 0 or len(alpha_all) == 0:
        # fallback conservative defaults
        return 1.5, 0.6
    return _robust_std(a_all), _robust_std(alpha_all)


def ctrv_predict_gaussian_xy_from_state(
    x, y, v, yaw, omega, delta_t,
    # Initial (state) uncertainty:
    sigma_pos=0.5, sigma_speed=0.5, sigma_yaw=0.05, sigma_omega=0.10,
    # Process noise:
    sigma_a=1.5, sigma_alpha=0.6,
    omega_eps=1e-3
):
    """
    CTRV + EKF linearization -> (x,y) ~ N(mu_xy, Sigma_xy)
    """
    mu = np.array([x, y, v, yaw, omega], dtype=float)
    P = np.diag([sigma_pos**2, sigma_pos**2, sigma_speed**2, sigma_yaw**2, sigma_omega**2])

    def f(s, dt):
        x, y, v, psi, om = s
        if abs(om) < omega_eps:
            x2 = x + v * dt * np.cos(psi)
            y2 = y + v * dt * np.sin(psi)
            psi2 = psi + om * dt
        else:
            th = psi + om * dt
            x2 = x + (v / om) * (np.sin(th) - np.sin(psi))
            y2 = y + (v / om) * (-np.cos(th) + np.cos(psi))
            psi2 = th
        return np.array([x2, y2, v, wrap_angle(psi2), om], dtype=float)

    def jacobian_F(s, dt):
        x, y, v, psi, om = s
        F = np.eye(5, dtype=float)

        if abs(om) < omega_eps:
            F[0, 2] = dt * np.cos(psi)
            F[0, 3] = -v * dt * np.sin(psi)
            F[1, 2] = dt * np.sin(psi)
            F[1, 3] =  v * dt * np.cos(psi)
            # Limit of the CTRV derivatives w.r.t. omega as omega -> 0
            F[0, 4] = -0.5 * v * dt**2 * np.sin(psi)
            F[1, 4] =  0.5 * v * dt**2 * np.cos(psi)
            F[3, 4] = dt
        else:
            th = psi + om * dt
            sin_t, cos_t = np.sin(th), np.cos(th)
            sin_p, cos_p = np.sin(psi), np.cos(psi)

            F[0, 2] = (sin_t - sin_p) / om
            F[1, 2] = (-cos_t + cos_p) / om

            F[0, 3] = (v / om) * (cos_t - cos_p)
            F[1, 3] = (v / om) * (sin_t - sin_p)

            F[0, 4] = v * ((dt * cos_t) / om - (sin_t - sin_p) / (om**2))
            F[1, 4] = v * ((dt * sin_t) / om - (-cos_t + cos_p) / (om**2))

            F[3, 4] = dt

        return F

    # Process noise: random longitudinal acceleration (sigma_a) and yaw acceleration
    # (sigma_alpha), constant over delta_t, mapped into the full state so that it
    # also affects the predicted position covariance.
    # Yaw acceleration also bends the path: lateral offset ~ v * alpha * dt^3 / 6.
    dt = delta_t
    psi_m = yaw + 0.5 * omega * dt          # mean heading over the interval
    c, s_ = np.cos(psi_m), np.sin(psi_m)
    G = np.array([
        [0.5 * dt**2 * c, -v * dt**3 / 6.0 * s_],
        [0.5 * dt**2 * s_,  v * dt**3 / 6.0 * c],
        [dt,               0.0],
        [0.0,              0.5 * dt**2],
        [0.0,              dt],
    ], dtype=float)
    Q = G @ np.diag([sigma_a**2, sigma_alpha**2]) @ G.T

    mu_pred = f(mu, delta_t)
    F = jacobian_F(mu, delta_t)
    P_pred = F @ P @ F.T + Q

    mu_xy = mu_pred[:2]#position
    Sigma_xy = P_pred[:2, :2]
    return mu_xy, Sigma_xy


def predict_gaussians_for_objects(objects, delta_t=1.0,
                                 sigma_pos=0.5, sigma_speed=0.5, sigma_yaw=0.05, sigma_omega=0.10,
                                 sigma_a=None, sigma_alpha=None,
                                 class_params=None):
    """Predict the (x, y) Gaussian of every object after delta_t seconds.

    class_params (optional, e.g. config ModelParameters().class_motion):
        {obj_type: {model, max_omega, sigma_pos, sigma_speed, sigma_yaw, sigma_omega,
                    sigma_a: (min, max), sigma_alpha: (min, max)}, 'default': {...}}
    When given, the initial uncertainty, the process noise (estimated per class and
    clipped to the class range) and the motion model ('CV' forces omega = 0) are
    chosen per object class. Without it the old global behaviour is used.
    """
    if class_params is None:
        if sigma_a is None or sigma_alpha is None:
            est_a, est_alpha = estimate_process_noise(objects)
            sigma_a = est_a if sigma_a is None else sigma_a
            sigma_alpha = est_alpha if sigma_alpha is None else sigma_alpha
    else:
        est_by_class = estimate_process_noise(objects, by_class=True)

    preds = {}
    for obj_id, seq in objects.items():
        if len(seq) < 1:
            continue
        seq = sorted(seq, key=lambda r: (r[0], r[1]))
        sec, nsec, x, y, v, yaw, omega, obj_type, _, _ = seq[-1]
        t_last = to_time_seconds(sec, nsec)

        if class_params is None:
            kw = dict(sigma_pos=sigma_pos, sigma_speed=sigma_speed, sigma_yaw=sigma_yaw,
                      sigma_omega=sigma_omega, sigma_a=sigma_a, sigma_alpha=sigma_alpha)
        else:
            cp = class_params.get(obj_type, class_params.get('default', {}))
            a_min, a_max = cp.get('sigma_a', (0.5, 3.0))
            al_min, al_max = cp.get('sigma_alpha', (0.1, 1.0))
            # estimated per class, else middle of the allowed range
            est_a, est_al = est_by_class.get(obj_type, (0.5 * (a_min + a_max), 0.5 * (al_min + al_max)))
            if not np.isfinite(est_a):
                est_a = 0.5 * (a_min + a_max)
            if not np.isfinite(est_al):
                est_al = 0.5 * (al_min + al_max)
            kw = dict(sigma_pos=cp.get('sigma_pos', sigma_pos),
                      sigma_speed=cp.get('sigma_speed', sigma_speed),
                      sigma_yaw=cp.get('sigma_yaw', sigma_yaw),
                      sigma_omega=cp.get('sigma_omega', sigma_omega),
                      sigma_a=float(np.clip(est_a, a_min, a_max)) if sigma_a is None else sigma_a,
                      sigma_alpha=float(np.clip(est_al, al_min, al_max)) if sigma_alpha is None else sigma_alpha)
            if cp.get('model', 'CTRV') == 'CV':
                omega = 0.0
            else:
                max_om = cp.get('max_omega', None)
                if max_om is not None:
                    omega = float(np.clip(omega, -max_om, max_om))

        mu_xy, Sigma_xy = ctrv_predict_gaussian_xy_from_state(
            x, y, v, yaw, omega, delta_t, **kw
        )

        preds[obj_id] = {
            "t_last": t_last,
            "type": obj_type,
            "mu": mu_xy.tolist(),
            "Sigma": Sigma_xy.tolist(),
            "params": {"sigma_a": kw["sigma_a"], "sigma_alpha": kw["sigma_alpha"]}
        }

    return preds


def gaussian_equation_str(mu, Sigma, delta_t):
    mx, my = mu
    S = np.array(Sigma, dtype=float)
    return (
        f"(x,y)_(t+{delta_t}) ~ N(mu, Sigma)\n"
        f"mu = [{mx:.6f}, {my:.6f}]^T\n"
        f"Sigma = [[{S[0,0]:.6f}, {S[0,1]:.6f}],\n"
        f"         [{S[1,0]:.6f}, {S[1,1]:.6f}]]"
    )


if __name__ == "__main__":
    objects = {
        183079: [
            (1765455671, 600000000, -31.70, 12.10, 0.55, -0.78, 0.45, "Car", 1.8, 4.5),
            (1765455671, 700000000, -31.61, 12.13, 0.57, -0.76, 0.47, "Car", 1.8, 4.5),
        ]
    }

    preds_1s = predict_gaussians_for_objects(objects, delta_t=1.0)
    p = preds_1s[183079]
    print(gaussian_equation_str(p["mu"], p["Sigma"], 1.0))

    preds_2s = predict_gaussians_for_objects(objects, delta_t=2.0)
    p2 = preds_2s[183079]
    print(gaussian_equation_str(p2["mu"], p2["Sigma"], 2.0))