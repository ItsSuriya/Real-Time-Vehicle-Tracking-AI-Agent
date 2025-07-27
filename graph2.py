import networkx as nx
import matplotlib.pyplot as plt

def visualize_tracking_path(G, tracked_path):
    labels = {}
    visit_count = {node: 0 for node in G.nodes}
    order = {}

    for i, node in enumerate(tracked_path):
        visit_count[node] += 1
        order[node, visit_count[node]] = i + 1

    for node in G.nodes:
        if visit_count[node] > 0:
            label_list = [str(order[node, count]) for count in range(1, visit_count[node] + 1)]
            labels[node] = f"{node} ({', '.join(label_list)})"
        else:
            labels[node] = str(node)

    visited_edges = list(zip(tracked_path, tracked_path[1:]))
    edge_colors = []
    edge_labels = {}
    edge_traversal_count = {}

    for u, v in visited_edges:
        edge_traversal_count[(u, v)] = edge_traversal_count.get((u, v), 0) + 1

    for u, v in G.edges:
        if (u, v) in edge_traversal_count:
            edge_colors.append('blue')
            count = edge_traversal_count[(u, v)]
            edge_labels[(u, v)] = f"{u}→{v} ({count}x)" if count > 1 else f"{u}→{v}"
        else:
            edge_colors.append('gray')

    # 🎯 Accurate layout like the image
    pos = {
        1: (2.5, 4.0),  # A
        2: (1.2, 3.2),  # B
        3: (3.7, 3.2),  # C
        4: (2.5, 2.6),  # D
        5: (0.5, 2.1),  # E
        6: (1.2, 1.2),  # F
        7: (4.5, 2.1),  # G
        8: (3.2, 1.2),  # H
    }

    plt.figure(figsize=(10, 8))
    node_colors = ['lightgreen' if visit_count[node] > 0 else 'lightblue' for node in G.nodes]

    nx.draw(G, pos, with_labels=True, labels=labels,
            node_color=node_colors, edge_color=edge_colors,
            node_size=2000, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Vehicle Tracking Path")
    plt.axis('off')
    plt.show()


# Create graph and add edges
G = nx.DiGraph()
G.add_edges_from([
    (1, 2), (1, 3),
    (2, 4), (2, 5),
    (3, 4), (3, 7),
    (4, 8),
    (7, 6), (7, 8),
    (6, 8)
])

# Example tracked path
tracked_path = [1, 3, 7, 6]

visualize_tracking_path(G, tracked_path)