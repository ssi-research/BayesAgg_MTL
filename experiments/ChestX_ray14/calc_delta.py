import numpy as np

chest_x_rays_target_dict = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Consolidation",
    3: "Edema",
    4: "Effusion",
    5: "Emphysema",
    6: "Fibrosis",
    7: "Hernia",
    8: "Infiltration",
    9: "Mass",
    10: "Nodule",
    11: "Pleural_Thickening",
    12: "Pneumonia",
    13: "Pneumothorax"
}

n_tasks = len(chest_x_rays_target_dict.keys())
BASE_val = np.array(
    [
        0.8056,
        0.8983,
        0.7790,
        0.9194,
        0.8939,
        0.6962,
        0.6846,
        0.7037,
        0.6857,
        0.6991,
        0.5879,
        0.6734,
        0.6477,
        0.7663,
    ]
)
BASE_test = np.array(
    [
        0.7543,
        0.8615,
        0.7132,
        0.8212,
        0.8224,
        0.6333,
        0.7357,
        0.7647,
        0.6830,
        0.6208,
        0.5894,
        0.6389,
        0.5710,
        0.7701,
    ]
)  # ours
# BASE_val = np.zeros(n_tasks) + 0.1
# BASE_test = np.zeros(n_tasks) + 0.1

SIGN = np.ones(n_tasks)

def delta_fn(a, dataset="val"):
    BASE = BASE_val if dataset == "val" else BASE_test
    return (((-1) ** SIGN) * (a - BASE) / BASE).mean() * 100.0  # *100 for percentage
