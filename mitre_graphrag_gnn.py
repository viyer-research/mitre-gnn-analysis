"""
GraphRAG+GNN Implementation for MITRE ATT&CK Evaluation
Combines Graph RAG with Graph Neural Networks for intelligent context selection
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network for learning entity importance from graph structure.
    Uses 2 layers of GCN with attention mechanism.
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128):
        """
        Args:
            input_dim: Dimension of input embeddings (from SentenceTransformer)
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output embeddings
        """
        super(GraphConvolutionalNetwork, self).__init__()
        
        # GCN layers for learning graph structure
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        # Attention mechanism for node importance
        self.attention_layer = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features (N, input_dim)
            edge_index: Graph edges (2, E)
            
        Returns:
            enhanced_embeddings: Learned node representations (N, output_dim)
            attention_weights: Node importance scores (N, 1)
        """
        # First fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Aggregate neighboring node information (simplified GCN)
        if edge_index.shape[1] > 0:
            # Get source and target nodes
            source, target = edge_index
            
            # Aggregate features from neighbors
            for _ in range(2):  # Two aggregation steps
                neighbor_features = x[source]
                x_agg = torch.zeros_like(x)
                x_agg.scatter_add_(0, target.unsqueeze(1).expand(-1, x.shape[1]), neighbor_features)
                
                # Combine with own features
                x = 0.7 * x + 0.3 * x_agg / (torch.bincount(target, minlength=x.shape[0]).float().unsqueeze(1) + 1)
        
        # Second fully connected layer
        x = self.fc2(x)
        x = self.relu(x)
        
        # Compute attention weights
        attention_weights = self.attention_layer(x)
        
        return x, attention_weights


class GraphRAGGNNProcessor:
    """
    Combines Graph RAG with GNN for intelligent context selection.
    Uses learned importance weights to select most relevant graph nodes.
    """
    
    def __init__(self, embedding_model, device: str = 'cpu', hidden_dim: int = 256, output_dim: int = 128):
        """
        Initialize GraphRAG+GNN processor.
        
        Args:
            embedding_model: SentenceTransformer model for embeddings
            device: 'cpu' or 'cuda'
            hidden_dim: Hidden dimension for GNN
            output_dim: Output dimension for GNN
        """
        self.embedding_model = embedding_model
        self.device = device
        self.gnn = GraphConvolutionalNetwork(
            input_dim=384,  # all-MiniLM-L6-v2 output dimension
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)
        
        self.gnn.eval()  # Use in evaluation mode
        logger.info(f"GraphRAG+GNN initialized on {device}")
    
    def prepare_graph_data(self, entities: List[Dict], relationships: List[Dict]) -> Tuple[Data, Dict]:
        """
        Convert entities and relationships to PyTorch Geometric Data format.
        
        Args:
            entities: List of entity dicts with 'id' and 'name' keys
            relationships: List of relationship dicts with 'source' and 'target' keys
            
        Returns:
            graph_data: PyTorch Geometric Data object
            entity_to_idx: Mapping from entity ID to node index
        """
        logger.info(f"Preparing graph data with {len(entities)} entities and {len(relationships)} relationships")
        
        # Create node embeddings
        entity_embeddings = []
        entity_to_idx = {}
        
        for idx, entity in enumerate(entities):
            try:
                entity_name = entity.get('name', entity.get('id', str(entity)))
                embedding = self.embedding_model.encode(entity_name, convert_to_tensor=False)
                entity_embeddings.append(embedding)
                entity_to_idx[entity['id']] = idx
            except Exception as e:
                logger.warning(f"Error encoding entity {entity}: {e}")
                continue
        
        # Convert to tensor
        node_features = torch.tensor(
            entity_embeddings,
            dtype=torch.float32,
            device=self.device
        )
        
        # Create edges from relationships
        edges = []
        for rel in relationships:
            source_id = rel.get('source')
            target_id = rel.get('target')
            
            if source_id in entity_to_idx and target_id in entity_to_idx:
                source_idx = entity_to_idx[source_id]
                target_idx = entity_to_idx[target_id]
                edges.append([source_idx, target_idx])
                # Add reverse edge for undirected graph
                edges.append([target_idx, source_idx])
        
        # Convert to tensor
        if edges:
            edge_index = torch.tensor(
                edges,
                dtype=torch.long,
                device=self.device
            ).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        
        # Create PyG Data object
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        logger.info(f"Graph data prepared: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        return graph_data, entity_to_idx
    
    def process_query(self, query: str, graph_data: Data, entity_to_idx: Dict, 
                     top_k: int = 10, alpha: float = 0.5) -> Tuple[List[str], np.ndarray]:
        """
        Process query using GNN to select most relevant context entities.
        
        Args:
            query: Query string
            graph_data: PyTorch Geometric Data object
            entity_to_idx: Entity ID to node index mapping
            top_k: Number of entities to select
            alpha: Weight for combining GNN attention (alpha) and query relevance (1-alpha)
            
        Returns:
            selected_entities: List of selected entity IDs
            relevance_scores: Relevance scores for selected entities
        """
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        query_embedding = query_embedding.to(self.device)
        
        # Get GNN outputs (enhanced embeddings + attention)
        with torch.no_grad():
            enhanced_embeddings, attention_weights = self.gnn(
                graph_data.x,
                graph_data.edge_index
            )
        
        # Compute cosine similarity between query and enhanced embeddings
        # Normalize for cosine similarity
        query_normalized = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        embeddings_normalized = torch.nn.functional.normalize(enhanced_embeddings, p=2, dim=1)
        
        query_relevance = torch.mm(query_normalized, embeddings_normalized.t()).squeeze()
        
        # Normalize scores to [0, 1]
        query_relevance_norm = (query_relevance - query_relevance.min()) / (query_relevance.max() - query_relevance.min() + 1e-8)
        attention_weights_norm = (attention_weights.squeeze() - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
        
        # Combine GNN attention with query relevance
        combined_scores = (alpha * attention_weights_norm + 
                          (1 - alpha) * query_relevance_norm)
        
        # Select top-K entities
        top_k = min(top_k, graph_data.num_nodes)
        top_k_indices = torch.topk(combined_scores, top_k).indices.cpu().numpy()
        
        # Convert back to entity IDs
        idx_to_entity = {v: k for k, v in entity_to_idx.items()}
        selected_entities = [idx_to_entity[idx] for idx in top_k_indices]
        
        relevance_scores = combined_scores[top_k_indices].cpu().numpy()
        
        logger.info(f"Selected {len(selected_entities)} entities for query: {query[:50]}...")
        
        return selected_entities, relevance_scores


class GraphRAGGNNEvaluator:
    """
    Evaluation wrapper for GraphRAG+GNN approach.
    Generates responses using GNN-selected context.
    """
    
    def __init__(self, adb_client, embedding_model, gnn_processor, llm_client):
        """
        Initialize evaluator.
        
        Args:
            adb_client: ArangoDB client
            embedding_model: SentenceTransformer model
            gnn_processor: GraphRAGGNNProcessor instance
            llm_client: Ollama LLM client
        """
        self.adb = adb_client
        self.embedding_model = embedding_model
        self.gnn_processor = gnn_processor
        self.llm = llm_client
        self.db = adb_client.db('MITRE2kg')
        
        # Cache graph data
        self._graph_data = None
        self._entity_to_idx = None
    
    def _load_graph_data(self):
        """Load and cache graph data."""
        if self._graph_data is None:
            logger.info("Loading graph data...")
            
            # Get all entities
            entities_cursor = self.db.aql.execute(
                'FOR v IN MITRE2kg RETURN {id: v._key, name: v.name}'
            )
            entities = list(entities_cursor)
            
            # Get all relationships
            relationships_cursor = self.db.aql.execute(
                'FOR e IN MITRE2kg_edges RETURN {source: e._from, target: e._to}'
            )
            relationships = list(relationships_cursor)
            
            # Prepare graph
            self._graph_data, self._entity_to_idx = self.gnn_processor.prepare_graph_data(
                entities,
                relationships
            )
    
    def _get_entity_details(self, entity_id: str) -> Dict:
        """Get entity details from database."""
        try:
            entity = self.db.collection('MITRE2kg').get(entity_id)
            return entity or {}
        except:
            return {}
    
    def _build_context(self, entity_ids: List[str]) -> str:
        """Build context string from selected entities."""
        context_parts = []
        
        for entity_id in entity_ids:
            entity = self._get_entity_details(entity_id)
            if entity:
                context_parts.append(f"- {entity.get('name', entity_id)}: {entity.get('description', '')}")
        
        return "\n".join(context_parts)
    
    def evaluate_query(self, query: str) -> Dict:
        """
        Evaluate query using GraphRAG+GNN approach.
        
        Args:
            query: Query string
            
        Returns:
            Result dict with response, metrics, etc.
        """
        start_time = time.time()
        
        # Load graph if not already loaded
        self._load_graph_data()
        
        # Process with GNN
        selected_entities, relevance_scores = self.gnn_processor.process_query(
            query,
            self._graph_data,
            self._entity_to_idx,
            top_k=10,
            alpha=0.5
        )
        
        # Build context
        context = self._build_context(selected_entities)
        
        # Generate response with LLM
        prompt = f"""Using the following context about cybersecurity threats:

{context}

Please answer this question: {query}

Provide a comprehensive answer based on the context provided."""
        
        response = self.llm.generate(prompt, max_tokens=500)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'approach': 'GraphRAG+GNN',
            'query': query,
            'response': response,
            'selected_entities': selected_entities,
            'relevance_scores': relevance_scores.tolist(),
            'context_quality': float(relevance_scores.mean()),
            'latency_ms': latency_ms,
            'tokens': len(response.split()),
            'graph_coverage': len(selected_entities) / max(self._graph_data.num_nodes, 1),
        }


# Example usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize GNN processor
    gnn_processor = GraphRAGGNNProcessor(
        embedding_model,
        device='cpu',
        hidden_dim=256,
        output_dim=128
    )
    
    # Test with sample data
    sample_entities = [
        {'id': 'e1', 'name': 'Phishing Attack'},
        {'id': 'e2', 'name': 'Credential Theft'},
        {'id': 'e3', 'name': 'Keylogger'},
    ]
    
    sample_relationships = [
        {'source': 'e1', 'target': 'e2'},
        {'source': 'e2', 'target': 'e3'},
    ]
    
    # Prepare graph
    graph_data, entity_to_idx = gnn_processor.prepare_graph_data(
        sample_entities,
        sample_relationships
    )
    
    # Test query
    query = "What are credential theft techniques?"
    selected, scores = gnn_processor.process_query(query, graph_data, entity_to_idx)
    
    print(f"Selected entities: {selected}")
    print(f"Scores: {scores}")
