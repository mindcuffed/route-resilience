import numpy as np

from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from skimage.measure import label
from skimage.morphology import closing, disk, skeletonize


INPUT_PATH = r"outputs\predicted_tiled_mask.png"
OUTPUT_PATH = r"outputs\reconstruction_repaired.png"

MIN_COMPONENT_SIZE = 100
MAX_GAP = 120
MAX_CONNECTIONS = 100
DIRECTION_THRESHOLD = 0.20
LOOK_AHEAD = 15
LINE_WIDTH = 3


def filter_small_components(mask):
    labels = label(
        mask,
        connectivity=2
    )

    filtered = np.zeros_like(mask)

    component_ids, sizes = np.unique(
        labels,
        return_counts=True
    )

    for component_id, size in zip(
        component_ids,
        sizes
    ):
        if component_id != 0 and size >= MIN_COMPONENT_SIZE:
            filtered[labels == component_id] = True

    return filtered


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
    start_y = y
    start_x = x

    previous = None
    current = (y, x)

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
            start_x - current[1],
            start_y - current[0]
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

    skeleton_points = np.argwhere(skeleton)

    for y, x in skeleton_points:
        neighbours = get_neighbours(
            skeleton,
            int(y),
            int(x)
        )

        if len(neighbours) != 1:
            continue

        component_id = labels[y, x]

        if component_id == 0:
            continue

        direction = trace_direction(
            skeleton,
            int(y),
            int(x)
        )

        if direction is not None:
            endpoint_data.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "component": int(component_id),
                    "direction": direction
                }
            )

    return endpoint_data


def find_best_connection(endpoint_data):
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

    best_connection = None
    best_score = -float("inf")

    for first_index, second_index in candidate_pairs:
        first = endpoint_data[first_index]
        second = endpoint_data[second_index]

        if first["component"] == second["component"]:
            continue

        difference = np.array(
            [
                second["x"] - first["x"],
                second["y"] - first["y"]
            ],
            dtype=float
        )

        distance = np.linalg.norm(difference)

        if distance == 0:
            continue

        connection_direction = difference / distance

        alignment_first = np.dot(
            first["direction"],
            connection_direction
        )

        alignment_second = np.dot(
            second["direction"],
            -connection_direction
        )

        if (
            alignment_first < DIRECTION_THRESHOLD
            or alignment_second < DIRECTION_THRESHOLD
        ):
            continue

        score = (
            alignment_first
            + alignment_second
            - distance / MAX_GAP
        )

        if score > best_score:
            best_score = score
            best_connection = (
                (first["x"], first["y"]),
                (second["x"], second["y"]),
                distance
            )

    return best_connection


def main():
    mask = np.asarray(
        Image.open(INPUT_PATH).convert("L")
    ) > 0

    original_pixels = int(mask.sum())

    mask = filter_small_components(mask)

    mask = closing(
        mask,
        disk(2)
    )

    labels = label(
        mask,
        connectivity=2
    )

    components_before = int(labels.max())
    connections_created = 0

    for _ in range(MAX_CONNECTIONS):
        labels = label(
            mask,
            connectivity=2
        )

        endpoint_data = get_endpoint_data(
            mask,
            labels
        )

        connection = find_best_connection(
            endpoint_data
        )

        if connection is None:
            print("No more valid connections found.")
            break

        first, second, distance = connection

        repaired_image = Image.fromarray(
            (mask.astype(np.uint8) * 255),
            mode="L"
        )

        draw = ImageDraw.Draw(
            repaired_image
        )

        draw.line(
            [first, second],
            fill=255,
            width=LINE_WIDTH
        )

        mask = np.asarray(
            repaired_image
        ) > 0

        connections_created += 1

        if connections_created % 10 == 0:
            print(
                f"Connections created: "
                f"{connections_created}"
            )

    components_after = int(
        label(
            mask,
            connectivity=2
        ).max()
    )

    Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    ).save(OUTPUT_PATH)

    print()
    print("Original predicted pixels:", original_pixels)
    print("Repaired predicted pixels:", int(mask.sum()))
    print("Components before repair:", components_before)
    print("Components after repair:", components_after)
    print("Connections created:", connections_created)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()