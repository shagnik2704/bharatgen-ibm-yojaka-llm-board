"""
Knowledge Graph Builder - Per-PDF
==================================
Builds deep hierarchical knowledge graphs.
"""

import logging
from typing import List
import networkx as nx
# Make sure to import your HierarchicalChunk properly based on your project structure
# from utils.hierarchical_chunker import HierarchicalChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_hierarchical_kg(chunks: List[any]) -> nx.DiGraph:
    logger.info("Building deep hierarchical knowledge graph...")
    G = nx.DiGraph()
    
    if not chunks:
        return G
        
    document_id = chunks[0].document_id
    doc_node = f"doc:{document_id}"
    G.add_node(doc_node, type="document", id=document_id)
    
    for chunk in chunks:
        node_attrs = chunk.to_dict()
        node_attrs["type"] = "chunk"
        G.add_node(chunk.chunk_id, **node_attrs)
        
        # Link Root to Document
        if chunk.level == 1:
            G.add_edge(doc_node, chunk.chunk_id, relation="HAS_ROOT")
            
        # Link Child to Parent strictly (e.g. 1.4 -> 1.4.1)
        if chunk.parent_id:
            G.add_edge(chunk.parent_id, chunk.chunk_id, relation="CONTAINS_SUBSECTION")

    # REMOVED the NEXT_SECTION loop so that graph.predecessors() strictly returns Parents
    # and graph.successors() strictly returns Children in the Retriever.
            
    logger.info(f"✓ Knowledge Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G