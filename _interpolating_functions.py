import numpy as np


def interpolate_grid(w_res, w_points, neighbourhood=None,
                     eval_func=None, **kwargs):

    if neighbourhood is None:
        neighbourhood = ball_neighbourhood

    if eval_func is None:
        eval_func = weighted_arithm_mean

    return interpolate_grid_points(w_res, w_points,
                                   neighbourhood, eval_func,
                                   **kwargs)


def interpolate_grid_points(w_res, w_points,
                            neighbourhood, eval_func,
                            **kwargs):
    ws_res, wa_res = w_res
    data = np.zeros((len(wa_res), len(ws_res)))
    for i in range(len(ws_res)):
        for j in range(len(wa_res)):
            grid_point = np.array([ws_res[i], wa_res[j]])
            dist, mask = neighbourhood(
                w_points.points[:, :2] - grid_point, **kwargs)
            data[j, i] = eval_func(
                w_points.points[mask], dist[mask],
                w_points.weights[mask], **kwargs)

    print(data)

    return data


def ball_neighbourhood(vec, **kwargs):
    radius = kwargs.get('radius', 1)
    vec[:, 1] = np.deg2rad(vec[:, 1])
    distance = np.linalg.norm(vec, axis=1)
    return distance, distance <= radius


def weighted_arithm_mean(points, weights, dist, **kwargs):
    alpha = kwargs.get('alpha', 1)
    beta = kwargs.get('beta', 1)
    weights = gauss_potential(dist, weights, alpha, beta)
    scal_fac = kwargs.get('s', 1)
    return scal_fac * weights @ points[:, 2] / len(weights)


def gauss_potential(dist, weights, alpha, beta):
    return beta * np.exp(-alpha * weights * dist)
