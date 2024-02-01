import numpy as np

utkface_target_dict = {
    0: "age",
    1: "gender",
    2: "race"
}

n_tasks = len(utkface_target_dict.keys())
BASE_val = np.array(
    [
        0.1276,
        0.9197,
        0.8456
    ]
)
BASE_test = np.array(
    [
        0.1400,
        0.9232,
        0.8242
    ]
)

SIGN = np.array([0, 1, 1])


def delta_fn(a, dataset="val"):
    BASE = BASE_val if dataset == "val" else BASE_test
    return (((-1) ** SIGN) * (a - BASE) / BASE).mean() * 100.0  # *100 for percentage
