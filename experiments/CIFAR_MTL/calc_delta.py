import numpy as np

cifar100_target_dict = {
    0: "aquatic mammals",
    1: "fish",
    2: "flowers",
    3: "food containers",
    4: "fruit and vegetables",
    5: "household electrical devices",
    6: "household furniture",
    7: "insects",
    8: "large carnivores",
    9: "large man-made outdoor things",
    10: "large natural outdoor scenes",
    11: "large omnivores and herbivores",
    12: "medium-sized mammals",
    13: "non-insect invertebrates",
    14: "people",
    15: "reptiles",
    16: "small mammals",
    17: "trees",
    18: "vehicles 1",
    19: "vehicles 2",
}


n_tasks = len(cifar100_target_dict.keys())
BASE = np.array(
    [

    ]
)  # ours

SIGN = np.array([1] * n_tasks)
KK = np.ones(n_tasks) * -1


def delta_fn(a):
    return (KK ** SIGN * (a - BASE) / BASE).mean() * 100.0  # *100 for percentage


if __name__ == "__main__":
    results = dict(
        stl=np.array(
            [

            ]
        ),  # sanity
        ours=np.array(
            [

            ]
        ),
    )

    for k, v in results.items():
        print(f"{k}: {delta_fn(v):.3f}")
    pass
