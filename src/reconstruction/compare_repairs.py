import csv

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from skimage.measure import label
from skimage.morphology import closing, disk

from src.reconstruction.gap_repair import (
    INPUT_MASK,
    INPUT_PROBABILITY,
    MAX_CONNECTIONS,
    MAX_GAP,
    MIN_COMPONENT_SIZE,
    draw_connection,
    filter_small_components,
    find_best_connection,
    get_endpoint_data,
)


OUTPUT_CSV = r"outputs\repair_comparison.csv"


def prepare_mask():
    mask = np.asarray(
        Image.open(INPUT_MASK).convert("L")
    ) > 0

    mask = filter_small_components(mask)
    mask = closing(mask, disk(2))

    return mask


def component_count(mask):
    return int(
        label(
            mask,
            connectivity=2
        ).max()
    )


def nearest_connection(endpoint_data):
    if len(endpoint_data) < 2:
        return None

    points = np.array(
        [
            [item["x"], item["y"]]
            for item in endpoint_data
        ],
        dtype=float
    )

    tree = cKDTree(points)
    pairs = tree.query_pairs(MAX_GAP)

    best = None

    for first_index, second_index in pairs:
        first = endpoint_data[first_index]
        second = endpoint_data[second_index]

        if first["component"] == second["component"]:
            continue

        distance = np.linalg.norm(
            np.array(
                [
                    second["x"] - first["x"],
                    second["y"] - first["y"]
                ],
                dtype=float
            )
        )

        if best is None or distance < best["distance"]:
           best = {
    "first_x": first["x"],
    "first_y": first["y"],
    "second_x": second["x"],
    "second_y": second["y"],
    "distance": float(distance)
}

    return best


def run_distance_repair(original_mask):
    mask = original_mask.copy()
    connections = 0

    for _ in range(MAX_CONNECTIONS):
        labels = label(
            mask,
            connectivity=2
        )

        endpoints = get_endpoint_data(
            mask,
            labels
        )

        connection = nearest_connection(endpoints)

        if connection is None:
            break

        mask = draw_connection(
            mask,
            connection
        )

        connections += 1

    return mask, connections


def run_guided_repair(original_mask, probability):
    mask = original_mask.copy()
    connections = 0

    for _ in range(MAX_CONNECTIONS):
        labels = label(
            mask,
            connectivity=2
        )

        endpoints = get_endpoint_data(
            mask,
            labels
        )

        connection = find_best_connection(
            endpoints,
            labels,
            probability
        )

        if connection is None:
            break

        mask = draw_connection(
            mask,
            connection
        )

        connections += 1

    return mask, connections


def save_mask(mask, path):
    Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    ).save(path)


def main():
    original_mask = prepare_mask()

    try:
        probability = np.load(INPUT_PROBABILITY)
    except FileNotFoundError:
        probability = original_mask.astype(np.float32)

    no_repair_mask = original_mask.copy()

    distance_mask, distance_connections = run_distance_repair(
        original_mask
    )

    guided_mask, guided_connections = run_guided_repair(
        original_mask,
        probability
    )

    save_mask(
        no_repair_mask,
        r"outputs\comparison_no_repair.png"
    )

    save_mask(
        distance_mask,
        r"outputs\comparison_distance_repair.png"
    )

    save_mask(
        guided_mask,
        r"outputs\comparison_guided_repair.png"
    )

    results = [
        {
            "method": "No repair",
            "components": component_count(no_repair_mask),
            "road_pixels": int(no_repair_mask.sum()),
            "connections": 0
        },
        {
            "method": "Distance-only repair",
            "components": component_count(distance_mask),
            "road_pixels": int(distance_mask.sum()),
            "connections": distance_connections
        },
        {
            "method": "Resilience-guided repair",
            "components": component_count(guided_mask),
            "road_pixels": int(guided_mask.sum()),
            "connections": guided_connections
        }
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "components",
                "road_pixels",
                "connections"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("Repair comparison:")

    for result in results:
        print(
            f"{result['method']} | "
            f"Components: {result['components']} | "
            f"Road pixels: {result['road_pixels']} | "
            f"Connections: {result['connections']}"
        )

    print()
    print("Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()