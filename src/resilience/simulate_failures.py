import csv
import math
import random

import networkx as nx
import numpy as np

from PIL import Image
from skimage.morphology import skeletonize


MASK_PATH = r"outputs\reconstruction_repaired.png"
OUTPUT_PATH = r"outputs\resilience_results.csv"


def build_pixel_graph(mask):
    skeleton = skeletonize(mask)
    graph = nx.Graph()
    pixels = np.argwhere(skeleton)

    for y, x in pixels:
        graph.add_node((int(y), int(x)))

    for y, x in pixels:
        y = int(y)
        x = int(x)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue

                ny = y + dy
                nx_pixel = x + dx

                if (
                    0 <= ny < skeleton.shape[0]
                    and 0 <= nx_pixel < skeleton.shape[1]
                    and skeleton[ny, nx_pixel]
                ):
                    graph.add_edge(
                        (y, x),
                        (ny, nx_pixel),
                        weight=math.hypot(dx, dy)
                    )

    return graph


def build_segment_graph(pixel_graph):
    segment_graph = nx.Graph()

    key_nodes = {
        node for node, degree in pixel_graph.degree()
        if degree != 2
    }

    segment_graph.add_nodes_from(key_nodes)
    visited_edges = set()

    for start in key_nodes:
        for first in pixel_graph.neighbors(start):
            edge_key = frozenset((start, first))

            if edge_key in visited_edges:
                continue

            visited_edges.add(edge_key)

            previous = start
            current = first
            length = math.dist(start, first)

            while current not in key_nodes:
                next_nodes = [
                    node for node in pixel_graph.neighbors(current)
                    if node != previous
                ]

                if not next_nodes:
                    break

                next_node = next_nodes[0]
                visited_edges.add(
                    frozenset((current, next_node))
                )

                length += math.dist(current, next_node)
                previous = current
                current = next_node

            if current in key_nodes and current != start:
                segment_graph.add_edge(
                    start,
                    current,
                    weight=length
                )

    return segment_graph


def largest_component_ratio(graph, original_nodes):
    components = list(nx.connected_components(graph))

    if not components:
        return 0.0

    largest_size = max(len(component) for component in components)
    return largest_size / original_nodes


def pair_reachability(graph, pairs):
    component_ids = {}

    for number, component in enumerate(
        nx.connected_components(graph)
    ):
        for node in component:
            component_ids[node] = number

    reachable = 0

    for source, target in pairs:
        if (
            source in component_ids
            and target in component_ids
            and component_ids[source] == component_ids[target]
        ):
            reachable += 1

    return reachable / len(pairs)


def main():
    mask = np.asarray(
        Image.open(MASK_PATH).convert("L")
    ) > 0

    pixel_graph = build_pixel_graph(mask)
    graph = build_segment_graph(pixel_graph)

    if graph.number_of_edges() == 0:
        print("No road segments were created.")
        return

    original_nodes = graph.number_of_nodes()
    edges = list(graph.edges())

    largest_component = max(
        nx.connected_components(graph),
        key=len
    )

    nodes = list(largest_component)
    generator = random.Random(42)

    pairs = [
        (
            generator.choice(nodes),
            generator.choice(nodes)
        )
        for _ in range(200)
    ]

    centrality = nx.betweenness_centrality(
        graph,
        weight="weight",
        normalized=True
    )

    ranked_edges = sorted(
        edges,
        key=lambda edge: (
            centrality[edge[0]]
            + centrality[edge[1]]
        ),
        reverse=True
    )

    results = []

    for failure_rate in (0.10, 0.20, 0.30):
        remove_count = max(
            1,
            int(len(edges) * failure_rate)
        )

        random_edges = generator.sample(
            edges,
            min(remove_count, len(edges))
        )

        targeted_edges = ranked_edges[:remove_count]

        for scenario, failed_edges in (
            ("Random Failure", random_edges),
            ("Targeted Failure", targeted_edges)
        ):
            damaged_graph = graph.copy()
            damaged_graph.remove_edges_from(failed_edges)

            largest_ratio = largest_component_ratio(
                damaged_graph,
                original_nodes
            )

            reachability = pair_reachability(
                damaged_graph,
                pairs
            )

            resilience = (
                largest_ratio + reachability
            ) / 2

            result = {
                "scenario": scenario,
                "failure_rate": failure_rate,
                "largest_component_ratio": largest_ratio,
                "pair_reachability": reachability,
                "resilience_score": resilience
            }

            results.append(result)

            print(
                scenario,
                "| Failure:",
                f"{failure_rate:.0%}",
                "| Largest component:",
                f"{largest_ratio:.3f}",
                "| Reachability:",
                f"{reachability:.3f}",
                "| Resilience:",
                f"{resilience:.3f}"
            )

    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "failure_rate",
                "largest_component_ratio",
                "pair_reachability",
                "resilience_score"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("Segment nodes:", original_nodes)
    print("Road segments:", len(edges))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()