import numpy as np
import networkx as nx

from PIL import Image
from skimage.morphology import skeletonize


mask_path = r"outputs\reconstruction_repaired.png"
output_path = r"outputs\ground_truth_graph.png"

mask = np.asarray(
    Image.open(mask_path).convert("L")
) > 0

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
                    weight=float(np.hypot(dx, dy))
                )

components = list(nx.connected_components(graph))
largest_component = max(components, key=len) if components else set()

junctions = [
    node for node, degree in graph.degree()
    if degree >= 3
]

if len(graph) > 2:
    centrality = nx.betweenness_centrality(
        graph,
        k=min(20, len(graph) - 1),
        weight="weight",
        seed=42
    )
else:
    centrality = {}

critical_nodes = sorted(
    centrality,
    key=centrality.get,
    reverse=True
)[:20]

visual = np.zeros(
    (skeleton.shape[0], skeleton.shape[1], 3),
    dtype=np.uint8
)

visual[skeleton] = [255, 255, 255]

for y, x in junctions:
    visual[y, x] = [255, 180, 0]

for y, x in critical_nodes:
    visual[y, x] = [255, 0, 0]

Image.fromarray(visual).save(output_path)

print("Graph nodes:", graph.number_of_nodes())
print("Graph edges:", graph.number_of_edges())
print("Connected components:", len(components))
print("Largest component nodes:", len(largest_component))
print("Junction pixels:", len(junctions))
print("Critical nodes:", len(critical_nodes))
print("Saved:", output_path)