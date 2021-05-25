import numpy as np


def interpolate_grid(w_res, w_points, neighbourhood=None,
                     eval_func=None, radius=1, alpha=1,
                     beta=1, **kwargs):
    ws_res, wa_res = w_res
    ws_res, wa_res = np.meshgrid(ws_res, np.deg2rad(wa_res),
                                 indexing='ij')

    if neighbourhood is None:
        neighbourhood = ball_neighbourhood

    kwargs["radius"] = radius

    if eval_func is None:
        eval_func = gauss_potential

    kwargs["alpha"] = alpha
    kwargs["beta"] = beta

    return interpolate_grid_points(ws_res, wa_res, w_points,
                                   neighbourhood, eval_func,
                                   **kwargs)


def interpolate_grid_points(ws_res, wa_res, w_points,
                            neighbourhood, eval_func,
                            **kwargs):
    data = np.zeros((len(wa_res), len(ws_res)))
    for i in range(len(ws_res)):
        for j in range(len(wa_res)):
            grid_point = np.array([ws_res[j, i], wa_res[j, i]])
            dist, mask = neighbourhood(
                w_points.points[:, :2] - grid_point, kwargs["radius"])
            dist = dist[mask]
            weights = w_points.weights[mask]
            data[j, i] = eval_func(dist, weights, **kwargs)

    return data


def ball_neighbourhood(vec, radius):
    distance = np.sum(np.square(vec, axis=1), axis=1)
    return distance, distance <= radius


def gauss_potential(dist, weight, **kwargs):
    alpha = kwargs.get("alpha")
    beta = kwargs.get("beta")
    return np.sum(beta * np.exp(-alpha * np.norm(weight * dist)))
