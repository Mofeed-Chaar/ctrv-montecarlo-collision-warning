

import numpy as np

from process import get_shifted_coordinate
from trajectory import predict_gaussians_for_objects


def inside_region_px(point_px, region_px):
    """
    region_px = (x1, y1, x2, y2)
    (x1, y1): top-left
    (x2, y2): bottom-right
    """
    u, v = point_px
    x1, y1, x2, y2 = region_px

    return x1 <= u <= x2 and y1 <= v <= y2


def rectangle_corners(center, yaw, length, width):
    center = np.array(center, dtype=float)

    half_l = length / 2.0
    half_w = width / 2.0

    local = np.array([
        [ half_l,  half_w],
        [ half_l, -half_w],
        [-half_l, -half_w],
        [-half_l,  half_w]
    ])

    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])

    return local @ R.T + center


def rectangles_overlap(c1, yaw1, length1, width1,
                       c2, yaw2, length2, width2):
    """
    OBB collision detection using SAT.
    """

    rect1 = rectangle_corners(c1, yaw1, length1, width1)
    rect2 = rectangle_corners(c2, yaw2, length2, width2)

    def project(poly, axis):
        values = poly @ axis
        return np.min(values), np.max(values)

    axes = []

    for rect in (rect1, rect2):
        for i in range(4):
            edge = rect[(i + 1) % 4] - rect[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)

            if norm > 1e-9:
                axes.append(axis / norm)

    for axis in axes:
        min1, max1 = project(rect1, axis)
        min2, max2 = project(rect2, axis)

        if max1 < min2 or max2 < min1:
            return False

    return True


def center_distance_risk(p1, p2, sigma=2.0):
    """
    Continuous distance-based risk.
    Smaller distance -> higher risk.

    risk = exp(-distance / sigma)
    """
    d = np.linalg.norm(np.array(p1) - np.array(p2))
    return float(np.exp(-d / sigma))


def classify_risk(probability):
    """
    Risk level based on collision probability.
    """
    if probability >= 0.20:
        return "CRITICAL"
    elif probability >= 0.10:
        return "HIGH"
    elif probability >= 0.05:
        return "MEDIUM"
    elif probability >= 0.01:
        return "LOW"
    else:
        return "SAFE"


def get_latest_states(objects):
    """
    Input format:
    (
        timestamp,
        nanosecond,
        x,
        y,
        v,
        yaw,
        velocity_ang,
        obj_type,
        width,
        height
    )
    """

    latest_states = {}

    for obj_id, seq in objects.items():
        seq = sorted(seq, key=lambda r: (r[0], r[1]))
        last = seq[-1]

        latest_states[obj_id] = {
            "id": obj_id,
            "x": last[2],
            "y": last[3],
            "v": last[4],
            "yaw": last[5],
            "omega": last[6],
            "type": last[7],
            "width": last[8],
            "height": last[9]
        }

    return latest_states


def monte_carlo_collision_pair_px_region(
    pred1,
    pred2,
    state1,
    state2,
    region_px=None,
    n_samples=100000,
    seed=42,
    distance_risk_sigma=2.0,
    safety_margin_length=0.0,
    safety_margin_width=0.0
):
    rng = np.random.default_rng(seed)

    mu1 = np.array(pred1["mu"], dtype=float)
    S1 = np.array(pred1["Sigma"], dtype=float)

    mu2 = np.array(pred2["mu"], dtype=float)
    S2 = np.array(pred2["Sigma"], dtype=float)

    samples1 = rng.multivariate_normal(mu1, S1, size=n_samples)
    samples2 = rng.multivariate_normal(mu2, S2, size=n_samples)

    region_count = 0
    collision_count = 0

    total_continuous_risk = 0.0
    max_continuous_risk = 0.0

    length1 = state1["height"] + safety_margin_length
    width1 = state1["width"] + safety_margin_width

    length2 = state2["height"] + safety_margin_length
    width2 = state2["width"] + safety_margin_width

    for p1, p2 in zip(samples1, samples2):

        if region_px is not None:
            p1_px = get_shifted_coordinate(p1)
            p2_px = get_shifted_coordinate(p2)

            inside1 = inside_region_px(p1_px, region_px)
            inside2 = inside_region_px(p2_px, region_px)

            if not (inside1 and inside2):
                continue

        region_count += 1

        continuous_risk = center_distance_risk(
            p1,
            p2,
            sigma=distance_risk_sigma
        )

        total_continuous_risk += continuous_risk
        max_continuous_risk = max(max_continuous_risk, continuous_risk)

        collision = rectangles_overlap(
            c1=p1,
            yaw1=state1["yaw"],
            length1=length1,
            width1=width1,

            c2=p2,
            yaw2=state2["yaw"],
            length2=length2,
            width2=width2
        )

        if collision:
            collision_count += 1

    collision_probability_global = collision_count / n_samples

    if region_count > 0:
        collision_probability_given_region = collision_count / region_count
        mean_continuous_risk = total_continuous_risk / region_count
    else:
        collision_probability_given_region = 0.0
        mean_continuous_risk = 0.0

    return {
        "vehicle_1": {
            "id": state1["id"],
            "type": state1["type"],
            "center_current": (state1["x"], state1["y"]),
            "yaw": state1["yaw"],
            "speed": state1["v"],
            "width": state1["width"],
            "height": state1["height"]
        },

        "vehicle_2": {
            "id": state2["id"],
            "type": state2["type"],
            "center_current": (state2["x"], state2["y"]),
            "yaw": state2["yaw"],
            "speed": state2["v"],
            "width": state2["width"],
            "height": state2["height"]
        },

        "collision_probability_global": collision_probability_global,

        "collision_probability_given_region":
            collision_probability_given_region,

        "mean_continuous_risk": mean_continuous_risk,

        "max_continuous_risk": max_continuous_risk,

        "risk_level": classify_risk(collision_probability_global),

        "num_total_samples": n_samples,
        "num_region_samples": region_count,
        "num_collision_samples": collision_count
    }


def monte_carlo_all_pairs_collision_px_region(
    objects,
    delta_t,
    region_px,
    n_samples=100000,
    seed=42,
    distance_risk_sigma=2.0,
    safety_margin_length=0.0,
    safety_margin_width=0.0,
    preds = None
):
    if preds is None:
        preds = predict_gaussians_for_objects(objects, delta_t=delta_t)
    latest_states = get_latest_states(objects)

    results = {}
    obj_ids = list(preds.keys())

    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):

            id1 = obj_ids[i]
            id2 = obj_ids[j]

            result = monte_carlo_collision_pair_px_region(
                pred1=preds[id1],
                pred2=preds[id2],
                state1=latest_states[id1],
                state2=latest_states[id2],
                region_px=region_px,
                n_samples=n_samples,
                seed=seed,
                distance_risk_sigma=distance_risk_sigma,
                safety_margin_length=safety_margin_length,
                safety_margin_width=safety_margin_width
            )

            results[(id1, id2)] = result

    return results


if __name__ == "__main__":

    objects = {
        1: [
            (1765455671, 600000000, 0.0, 0.0, 0.0, 0.0, 0.0, "Car", 1.8, 4.5),
            (1765455671, 700000000, 0.0, 0.0, 0.0, 0.0, 0.0, "Car", 1.8, 4.5),
        ],

        2: [
            (1765455671, 600000000, 1.5, 0.0, 0.0, 0.0, 0.0, "Car", 1.8, 4.5),
            (1765455671, 700000000, 1.5, 0.0, 0.0, 0.0, 0.0, "Car", 1.8, 4.5),
        ]
    }

    delta_t = 0.0

    # None = no zone restriction
    region_px = None

    preds = predict_gaussians_for_objects(objects, delta_t=delta_t)
    all_results = monte_carlo_all_pairs_collision_px_region(
        objects=objects,
        delta_t=delta_t,
        region_px=region_px,
        n_samples=10000,
        seed=42,
        distance_risk_sigma=2.0,
        safety_margin_length=0.0,
        safety_margin_width=0.0,
        preds=preds
    )

    print("\nPredictions:")
    for obj_id, pred in preds.items():
        print(f"ID {obj_id}:")
        print(pred)
        print()

    print("\nCollision / Risk Results:")

    for pair, result in all_results.items():
        v1 = result["vehicle_1"]
        v2 = result["vehicle_2"]

        print("========================================")
        print(f"Pair: {v1['id']} ({v1['type']})  vs  {v2['id']} ({v2['type']})")

        print(f"Vehicle 1 current center: {v1['center_current']}")
        print(f"Vehicle 2 current center: {v2['center_current']}")

        print(f"Vehicle 1 size: width={v1['width']}, height={v1['height']}")
        print(f"Vehicle 2 size: width={v2['width']}, height={v2['height']}")

        print()
        print(f"Global collision probability    : {result['collision_probability_global']:.6f}")
        print(f"Collision probability in region : {result['collision_probability_given_region']:.6f}")
        print(f"Mean continuous risk            : {result['mean_continuous_risk']:.6f}")
        print(f"Max continuous risk             : {result['max_continuous_risk']:.6f}")
        print(f"Risk level                      : {result['risk_level']}")

        print()
        print(f"Total samples     : {result['num_total_samples']}")
        print(f"Region samples    : {result['num_region_samples']}")
        print(f"Collision samples : {result['num_collision_samples']}")
        print("========================================")
