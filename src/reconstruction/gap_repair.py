import csv

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from skimage.measure import label
from skimage.morphology import closing, disk, remove_small_objects, skeletonize


INPUT_MASK = r"outputs\predicted_tiled_mask.png"
INPUT_PROBABILITY = r"outputs\predicted_tiled_probability.npy"

OUTPUT_MASK = r"outputs\reconstruction_repaired.png"
OUTPUT_CONNECTIONS = r"outputs\accepted_connections.csv"

MIN_COMPONENT_SIZE = 100
MAX_GAP = 80
MAX_CONNECTIONS = 100
MIN_REPAIR_SCORE = 0.55
MIN_PATH_CONFIDENCE = 0.30

DIRECTION_THRESHOLD = 0.20
LOOK_AHEAD = 15
LINE_WIDTH = 3


def load_probability(mask):
    try:
        probability = np.load(INPUT_PROBABILITY)

        if probability.shape != mask.shape:
            print("Probability map shape mismatch.")
            return mask.astype(np.float32)

        return np.clip(
            probability.astype(np.float32),
            0.0,
            1.0
        )

    except FileNotFoundError:
        print("Probability map not found.")
        return mask.astype(np.float32)


def filter_small_components(mask):
    return remove_small_objects(
        mask.astype(bool),
        max_size=MIN_COMPONENT_SIZE
    )


def get_neighbours(skeleton, y, x):
    height, width = skeleton.shape
    neighbours = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx = x + dx

            if (
                0 <= ny < height
                and 0 <= nx < width
                and skeleton[ny, nx]
            ):
                neighbours.append((ny, nx))

    return neighbours


def trace_direction(skeleton, y, x):
    start = (y, x)
    previous = None
    current = start

    for _ in range(LOOK_AHEAD):
        neighbours = get_neighbours(
            skeleton,
            current[0],
            current[1]
        )

        if previous is not None:
            neighbours = [
                point for point in neighbours
                if point != previous
            ]

        if not neighbours:
            break

        next_point = neighbours[0]
        previous = current
        current = next_point

    direction = np.array(
        [
            start[1] - current[1],
            start[0] - current[0]
        ],
        dtype=float
    )

    length = np.linalg.norm(direction)

    if length == 0:
        return None

    return direction / length


def get_endpoint_data(mask, labels):
    skeleton = skeletonize(mask)
    endpoint_data = []

    for y, x in np.argwhere(skeleton):
        y = int(y)
        x = int(x)

        neighbours = get_neighbours(
            skeleton,
            y,
            x
        )

        if len(neighbours) != 1:
            continue

        component_id = int(labels[y, x])

        if component_id == 0:
            continue

        direction = trace_direction(
            skeleton,
            y,
            x
        )

        if direction is not None:
            endpoint_data.append(
                {
                    "x": x,
                    "y": y,
                    "component": component_id,
                    "direction": direction
                }
            )

    return endpoint_data


def get_line_pixels(first, second, shape):
    x1, y1 = first
    x2, y2 = second

    points = np.linspace(
        0.0,
        1.0,
        max(
            int(np.linalg.norm(
                np.array(second) - np.array(first)
            )),
            2
        )
    )

    xs = np.rint(
        x1 + points * (x2 - x1)
    ).astype(int)

    ys = np.rint(
        y1 + points * (y2 - y1)
    ).astype(int)

    valid = (
        (xs >= 0)
        & (xs < shape[1])
        & (ys >= 0)
        & (ys < shape[0])
    )

    return ys[valid], xs[valid]


def calculate_connection_scores(
    first,
    second,
    labels,
    probability,
    largest_component_size
):
    difference = np.array(
        [
            second["x"] - first["x"],
            second["y"] - first["y"]
        ],
        dtype=float
    )

    distance = np.linalg.norm(difference)

    if distance == 0:
        return None

    connection_direction = difference / distance

    first_alignment = np.dot(
        first["direction"],
        connection_direction
    )

    second_alignment = np.dot(
        second["direction"],
        -connection_direction
    )

    if (
        first_alignment < DIRECTION_THRESHOLD
        or second_alignment < DIRECTION_THRESHOLD
    ):
        return None

    direction_score = np.clip(
        (first_alignment + second_alignment) / 2.0,
        0.0,
        1.0
    )

    distance_score = np.clip(
        1.0 - distance / MAX_GAP,
        0.0,
        1.0
    )

    rows, columns = get_line_pixels(
        (first["x"], first["y"]),
        (second["x"], second["y"]),
        probability.shape
    )

    if len(rows) == 0:
        return None

    confidence_score = float(
        probability[rows, columns].mean()
    )

    if confidence_score < MIN_PATH_CONFIDENCE:
        return None

    first_size = int(
        np.sum(labels == first["component"])
    )

    second_size = int(
        np.sum(labels == second["component"])
    )

    merged_size = first_size + second_size

    connectivity_score = np.clip(
        merged_size / max(largest_component_size, 1),
        0.0,
        1.0
    )

    total_score = (
        0.35 * direction_score
        + 0.25 * distance_score
        + 0.20 * confidence_score
        + 0.20 * connectivity_score
    )

    return {
        "first_x": first["x"],
        "first_y": first["y"],
        "second_x": second["x"],
        "second_y": second["y"],
        "distance": float(distance),
        "direction_score": float(direction_score),
        "distance_score": float(distance_score),
        "confidence_score": float(confidence_score),
        "connectivity_score": float(connectivity_score),
        "total_score": float(total_score)
    }


def find_best_connection(endpoint_data, labels, probability):
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
    candidate_pairs = tree.query_pairs(MAX_GAP)

    component_sizes = np.bincount(
        labels.ravel()
    )

    if len(component_sizes) <= 1:
        return None

    largest_component_size = int(
        component_sizes[1:].max()
    )

    best_connection = None

    for first_index, second_index in candidate_pairs:
        first = endpoint_data[first_index]
        second = endpoint_data[second_index]

        if first["component"] == second["component"]:
            continue

        candidate = calculate_connection_scores(
            first,
            second,
            labels,
            probability,
            largest_component_size
        )

        if candidate is None:
            continue

        if candidate["total_score"] < MIN_REPAIR_SCORE:
            continue

        if (
            best_connection is None
            or candidate["total_score"]
            > best_connection["total_score"]
        ):
            best_connection = candidate

    return best_connection


def draw_connection(mask, connection):
    image = Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    )

    drawing = ImageDraw.Draw(image)

    drawing.line(
        [
            (
                connection["first_x"],
                connection["first_y"]
            ),
            (
                connection["second_x"],
                connection["second_y"]
            )
        ],
        fill=255,
        width=LINE_WIDTH
    )

    return np.asarray(image) > 0


def main():
    mask = np.asarray(
        Image.open(INPUT_MASK).convert("L")
    ) > 0

    original_pixels = int(mask.sum())

    probability = load_probability(mask)

    mask = filter_small_components(mask)
    mask = closing(mask, disk(2))

    initial_labels = label(
        mask,
        connectivity=2
    )

    components_before = int(
        initial_labels.max()
    )

    connections = []

    for iteration in range(MAX_CONNECTIONS):
        labels = label(
            mask,
            connectivity=2
        )

        endpoint_data = get_endpoint_data(
            mask,
            labels
        )

        connection = find_best_connection(
            endpoint_data,
            labels,
            probability
        )

        if connection is None:
            print(
                "No more valid high-confidence connections found."
            )
            break

        mask = draw_connection(
            mask,
            connection
        )

        connections.append(connection)

        print(
            f"Connection {iteration + 1}: "
            f"score={connection['total_score']:.3f}, "
            f"confidence={connection['confidence_score']:.3f}, "
            f"distance={connection['distance']:.1f}"
        )

    final_labels = label(
        mask,
        connectivity=2
    )

    components_after = int(
        final_labels.max()
    )

    Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    ).save(OUTPUT_MASK)

    with open(
        OUTPUT_CONNECTIONS,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "first_x",
                "first_y",
                "second_x",
                "second_y",
                "distance",
                "direction_score",
                "distance_score",
                "confidence_score",
                "connectivity_score",
                "total_score"
            ]
        )

        writer.writeheader()
        writer.writerows(connections)

    print()
    print("Original predicted pixels:", original_pixels)
    print("Repaired predicted pixels:", int(mask.sum()))
    print("Components before repair:", components_before)
    print("Components after repair:", components_after)
    print("Connections created:", len(connections))
    print("Saved:", OUTPUT_MASK)
    print("Saved:", OUTPUT_CONNECTIONS)


if __name__ == "__main__":
    main()