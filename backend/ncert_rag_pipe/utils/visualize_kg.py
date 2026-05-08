"""
Knowledge Graph Visualizer
==========================
Reads a graph.pkl file and outputs a visual tree of the document structure.
"""

import pickle
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

def visualize_hierarchy(pkl_path: str, output_image: str = "kg_visualization.png"):
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find {pkl_path}")
        
    print(f"Loading graph from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
        
    # 1. Filter out sequential reading edges (like 'NEXT_SECTION') 
    # to only show the top-down Hierarchy tree (Parent -> Child)
    tree_G = nx.DiGraph()
    hierarchical_relations = ["HAS_ROOT", "CONTAINS_SUBSECTION", "CONTAINS", "DOCUMENT"]
    
    for u, v, edge_data in G.edges(data=True):
        if edge_data.get("relation") in hierarchical_relations:
            tree_G.add_edge(u, v)
            
    # Add nodes that might be isolated
    for n in G.nodes():
        if n not in tree_G:
            tree_G.add_node(n)
            
    # 2. Extract and format titles for labels
    labels = {}
    for node, data in G.nodes(data=True):
        # Fallback to the node ID if title is missing
        raw_title = data.get("title", str(node))
        # Wrap long titles with newlines so they fit in the graph
        words = raw_title.split()
        if len(words) > 4:
            formatted_title = " ".join(words[:4]) + "\n" + " ".join(words[4:8])
            if len(words) > 8:
                formatted_title += "..."
        else:
            formatted_title = raw_title
            
        labels[node] = formatted_title

    # 3. Compute layout
    plt.figure(figsize=(18, 12))
    
    # Try to use Graphviz for perfect top-down trees, otherwise use spring layout
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(tree_G, prog='dot')
    except ImportError:
        print("Graphviz not installed. Falling back to Spring layout...")
        # Use spring layout but adjust weights to separate parent/child
        pos = nx.spring_layout(tree_G, k=0.9, iterations=50)

    # 4. Draw the graph components
    # Draw Nodes
    nx.draw_networkx_nodes(
        tree_G, pos, 
        node_size=3500, 
        node_color='#a0c4ff', 
        edgecolors='black',
        alpha=0.9
    )
    
    # Draw Edges
    nx.draw_networkx_edges(
        tree_G, pos, 
        arrows=True, 
        arrowsize=20, 
        edge_color='#555555',
        width=2.0
    )
    
    # Draw Labels
    nx.draw_networkx_labels(
        tree_G, pos, 
        labels, 
        font_size=8, 
        font_weight='bold',
        font_family='sans-serif'
    )

    plt.title(f"Knowledge Graph Hierarchy: {pkl_path.parent.name}", fontsize=20, pad=20)
    plt.axis('off')
    
    # 5. Save Image
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a RAG Knowledge Graph")
    parser.add_argument("--pkl", "-p", required=True, help="Path to the graph.pkl file")
    parser.add_argument("--out", "-o", default="kg_visualization.png", help="Output image filename")
    args = parser.parse_args()
    
    visualize_hierarchy(args.pkl, args.out)