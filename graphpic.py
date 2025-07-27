import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph to handle multiple edges
G = nx.DiGraph()

# Add edges (you can customize the graph structure here)
edges = [ (1, 2),  # Camera 1 → Camera 2
    (1, 3),  # Camera 1 → Camera 3
    (2, 4),  # Camera 2 → Camera 3
    (2, 5),  # Camera 2 → Camera 4
    (3, 4)]
G.add_edges_from(edges)

# Tracked path (example: 1 → 2 → 4 → 2 → 1)
tracked_path = [1, 2, 4, 3]

# Prepare labels for each node with visit order
labels = {}
visit_count = {node: 0 for node in G.nodes}
order = {}

# Populate the visit order
for i, node in enumerate(tracked_path):
    visit_count[node] += 1
    order[node, visit_count[node]] = i + 1

# Generate node labels including visit order
for node in G.nodes:
    if visit_count[node] > 0:
        label_list = [str(order[node, count]) for count in range(1, visit_count[node] + 1)]
        labels[node] = f"{node} ({', '.join(label_list)})"
    else:
        labels[node] = str(node)

# Edge colors and labels
edge_colors = []
edge_labels = {}
visited_edges = list(zip(tracked_path, tracked_path[1:]))

# Dictionary to keep track of edge traversal counts
edge_traversal_count = {}

for u, v in visited_edges:
    if (u, v) not in edge_traversal_count:
        edge_traversal_count[(u, v)] = 1
    else:
        edge_traversal_count[(u, v)] += 1

for u, v in G.edges:
    if (u, v) in visited_edges or (v, u) in visited_edges:
        edge_colors.append('blue')
        # Count how many times the edge was traversed and label accordingly
        count = edge_traversal_count.get((u, v), 0)
        edge_labels[(u, v)] = f"{u}→{v} ({count}x)" if count > 1 else f"{u}→{v}"
    else:
        edge_colors.append('gray')

# Draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, labels=labels, node_color=['lightgreen' if visit_count[node] > 0 else 'lightgray' for node in G.nodes], edge_color=edge_colors, node_size=2000, font_size=12, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Vehicle Tracking Path")
plt.show()