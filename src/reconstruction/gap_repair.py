import numpy as np

from PIL import Image, ImageDraw
from skimage.measure import label
from skimage.morphology import skeletonize


MASK_PATH = r"data\processed\masks\AOI_2_Vegas_img1454.png"

ORIGINAL_OUTPUT = r"outputs\reconstruction_original.png"
OCCLUDED_OUTPUT = r"outputs\reconstruction_occluded.png"
REPAIRED_OUTPUT = r"outputs\reconstruction_repaired.png"


def get_neighbors(skeleton, y, x):
    neighbors = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx = x + dx

            if (
                0 <= ny < skeleton.shape[0]
                and 0 <= nx < skeleton.shape[1]
                and skeleton[ny, nx]
            ):
                neighbors.append((ny, nx))

    return neighbors


def find_endpoints(skeleton, component_labels):
    endpoints = []

    pixels = np.argwhere(skeleton)

    for y, x in pixels:
        y = int(y)
        x = int(x)

        neighbors = get_neighbors(skeleton, y, x)

        if len(neighbors) == 1:
            neighbor_y, neighbor_x = neighbors[0]

            direction = np.array(
                [
                    y - neighbor_y,
                    x - neighbor_x
                ],
                dtype=float
            )

            length = np.linalg.norm(direction)

            if length > 0:
                direction = direction / length

                endpoints.append(
                    (
                        (y, x),
                        direction,
                        int(component_labels[y, x])
                    )
                )

    return endpoints


def repair_gaps(mask, max_gap=250):
    skeleton = skeletonize(mask)

    component_labels = label(
        skeleton,
        connectivity=2
    )

    endpoints = find_endpoints(
        skeleton,
        component_labels
    )

    candidates = []

    for first in range(len(endpoints)):
        point_a, direction_a, component_a = endpoints[first]

        for second in range(first + 1, len(endpoints)):
            point_b, direction_b, component_b = endpoints[second]

            if component_a == component_b:
                continue

            vector = np.array(
                [
                    point_b[0] - point_a[0],
                    point_b[1] - point_a[1]
                ],
                dtype=float
            )

            distance = np.linalg.norm(vector)

            if distance == 0 or distance > max_gap:
                continue

            unit_vector = vector / distance

            alignment_a = np.dot(
                direction_a,
                unit_vector
            )

            alignment_b = np.dot(
                direction_b,
                -unit_vector
            )

            if alignment_a < 0.5 or alignment_b < 0.5:
                continue

            score = (
                (alignment_a + alignment_b) / 2
                - distance / max_gap
            )

            candidates.append(
                (
                    score,
                    first,
                    second,
                    point_a,
                    point_b,
                    distance
                )
            )

    candidates.sort(reverse=True)

    repaired = (mask * 255).astype(np.uint8)
    repaired_image = Image.fromarray(repaired)
    drawer = ImageDraw.Draw(repaired_image)

    used_endpoints = set()
    connections = []

    for (
        score,
        first,
        second,
        point_a,
        point_b,
        distance
    ) in candidates:
        if first in used_endpoints or second in used_endpoints:
            continue

        y1, x1 = point_a
        y2, x2 = point_b

        drawer.line(
            [(x1, y1), (x2, y2)],
            fill=255,
            width=5
        )

        used_endpoints.add(first)
        used_endpoints.add(second)

        connections.append(
            {
                "start": point_a,
                "end": point_b,
                "distance": distance,
                "score": score
            }
        )

    return np.asarray(repaired_image) > 0, connections


def main():
    original = np.asarray(
        Image.open(MASK_PATH).convert("L")
    ) > 0

    original_image = Image.fromarray(
        (original * 255).astype(np.uint8)
    )

    original_image.save(ORIGINAL_OUTPUT)

    occluded_image = original_image.copy()
    drawer = ImageDraw.Draw(occluded_image)

    drawer.ellipse(
        (100, 150, 500, 600),
        fill=0
    )

    occluded = np.asarray(occluded_image) > 0
    occluded_image.save(OCCLUDED_OUTPUT)

    before_components = label(
        skeletonize(occluded),
        connectivity=2
    ).max()

    repaired, connections = repair_gaps(occluded)

    repaired_image = Image.fromarray(
        (repaired * 255).astype(np.uint8)
    )

    repaired_image.save(REPAIRED_OUTPUT)

    after_components = label(
        skeletonize(repaired),
        connectivity=2
    ).max()

    print("Original road pixels:", int(original.sum()))
    print("Occluded road pixels:", int(occluded.sum()))
    print("Repaired road pixels:", int(repaired.sum()))
    print("Components before repair:", int(before_components))
    print("Components after repair:", int(after_components))
    print("Connections created:", len(connections))
    print("Saved:", REPAIRED_OUTPUT)


if __name__ == "__main__":
    main()