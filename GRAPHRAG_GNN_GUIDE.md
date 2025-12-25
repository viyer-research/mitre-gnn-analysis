# Comparing RAG vs Graph+LLM vs GraphRAG+GNN

## Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR CURRENT SETUP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PURE RAG                                                    │
│     Query → Embed → Semantic Search → Retrieved Docs → LLM     │
│     (Fast, simple, no graph context)                           │
│                                                                 │
│  2. GRAPH+LLM (Current)                                         │
│     Query → Embed → Graph Traversal → Context Building → LLM   │
│     (Slower, richer context from graph relationships)          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│              WHAT YOU CAN ADD: GraphRAG+GNN                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  3. GRAPHRAG+GNN (New - Microsoft Research approach)           │
│     Query → Embed → GNN Processing → Graph Encoding            │
│        → Enhanced Context → LLM                                │
│     (Most sophisticated, learns graph structure patterns)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences

### Pure RAG
- **What it does**: Retrieves semantically similar documents
- **Graph usage**: None
- **Intelligence**: Statistical similarity
- **Speed**: ⚡ Fast (30-40s)
- **Context**: Limited to top-K documents
- **Learnable**: No

### Graph+LLM (Your Current Implementation)
- **What it does**: Traverses knowledge graph relationships
- **Graph usage**: Direct traversal (BFS/DFS)
- **Intelligence**: Rule-based exploration
- **Speed**: ⚡⚡ Medium (40-60s)
- **Context**: All connected entities within hop limit
- **Learnable**: No

### GraphRAG+GNN (What to Add)
- **What it does**: Learns optimal graph traversal patterns
- **Graph usage**: Neural network learns from graph structure
- **Intelligence**: Machine learning on graph embeddings
- **Speed**: ⚡⚡⚡ Slower (60-120s)
- **Context**: Intelligently weighted context
- **Learnable**: Yes - learns importance weights

## Implementation Strategy

### Step 1: Install Required Libraries

```bash
pip install torch torch-geometric numpy scipy
# torch-geometric is PyTorch Geometric for GNN
# For CPU-only (no GPU):
pip install torch-geometric --no-cache-dir
```

### Step 2: Add GNN Module to Your Framework

Create `mitre_gnn_processor.py`:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data
import numpy as np

class GraphNeuralNetwork(nn.Module):
    """
    Graph Convolutional Network for learning entity importance
    from MITRE2kg graph structure
    """
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128):
        super(GraphNeuralNetwork, self).__init__()
        
        # GCN layers learn from graph structure
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        # Attention layer for context weighting
        self.attention = nn.Linear(output_dim, 1)
        
    def forward(self, x, edge_index):
        """
        x: Node features (embeddings)
        edge_index: Graph edges
        Returns: Enhanced node embeddings with attention weights
        """
        # First GCN layer with activation
        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        
        # Second GCN layer
        x = self.gcn2(x, edge_index)
        
        # Attention weights for context selection
        attention_weights = torch.sigmoid(self.attention(x))
        
        return x, attention_weights

class GraphRAGProcessor:
    """
    Combines Graph RAG with GNN for intelligent context selection
    """
    def __init__(self, embedding_model, device='cpu'):
        self.embedding_model = embedding_model
        self.device = device
        self.gnn = GraphNeuralNetwork().to(device)
        
    def prepare_graph_data(self, entities, relationships):
        """
        Convert entities and relationships to PyTorch Geometric Data
        """
        # Create node embeddings
        entity_embeddings = []
        entity_to_idx = {}
        
        for idx, entity in enumerate(entities):
            embedding = self.embedding_model.encode(entity['name'])
            entity_embeddings.append(embedding)
            entity_to_idx[entity['id']] = idx
        
        node_features = torch.tensor(
            entity_embeddings, 
            dtype=torch.float32
        ).to(self.device)
        
        # Create edges from relationships
        edges = []
        for rel in relationships:
            if rel['source'] in entity_to_idx and rel['target'] in entity_to_idx:
                edges.append([
                    entity_to_idx[rel['source']],
                    entity_to_idx[rel['target']]
                ])
        
        edge_index = torch.tensor(
            edges, 
            dtype=torch.long
        ).t().contiguous().to(self.device)
        
        # Create PyG Data object
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        return graph_data, entity_to_idx
    
    def process_query(self, query, graph_data, entity_to_idx):
        """
        Process query using GNN to select most relevant context
        """
        # Encode query
        query_embedding = torch.tensor(
            self.embedding_model.encode(query),
            dtype=torch.float32
        ).to(self.device)
        
        # Get GNN outputs (enhanced embeddings + attention)
        with torch.no_grad():
            enhanced_embeddings, attention_weights = self.gnn(
                graph_data.x, 
                graph_data.edge_index
            )
        
        # Score entities by relevance to query
        query_relevance = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            enhanced_embeddings,
            dim=1
        )
        
        # Combine GNN attention with query relevance
        combined_scores = (attention_weights.squeeze() * 0.4 + 
                          query_relevance * 0.6)
        
        # Select top-K entities
        top_k = 10
        top_indices = torch.topk(combined_scores, top_k).indices.cpu().numpy()
        
        # Convert back to entity names
        idx_to_entity = {v: k for k, v in entity_to_idx.items()}
        selected_entities = [idx_to_entity[idx] for idx in top_indices]
        
        return selected_entities, combined_scores[top_indices].cpu().numpy()
```

### Step 3: Create Integrated Evaluation

```python
# In mitre_integrated_evaluation.py, add:

class GraphRAGEvaluator:
    """
    Evaluates GraphRAG+GNN approach
    """
    def __init__(self, embedding_model, adb, llm_judge):
        self.embedding_model = embedding_model
        self.adb = adb
        self.llm_judge = llm_judge
        self.gnn_processor = GraphRAGProcessor(embedding_model)
        
    def evaluate_query(self, query):
        """
        Evaluate using GraphRAG+GNN approach
        """
        start_time = time.time()
        
        # Fetch all entities and relationships
        entities = self._get_all_entities()
        relationships = self._get_all_relationships()
        
        # Prepare graph for GNN
        graph_data, entity_to_idx = self.gnn_processor.prepare_graph_data(
            entities, 
            relationships
        )
        
        # Process with GNN
        selected_entities, relevance_scores = self.gnn_processor.process_query(
            query, 
            graph_data, 
            entity_to_idx
        )
        
        # Build context from selected entities
        context = self._build_context_from_entities(selected_entities)
        
        # Generate response
        response = self._generate_response(query, context)
        
        latency = (time.time() - start_time) * 1000
        
        return {
            'approach': 'GraphRAG+GNN',
            'response': response,
            'context_entities': selected_entities,
            'relevance_scores': relevance_scores.tolist(),
            'latency_ms': latency,
            'tokens': len(response.split())
        }
```

### Step 4: Update Comparison Framework

```python
class TripleEvaluator:
    """
    Compare all three approaches: RAG vs Graph+LLM vs GraphRAG+GNN
    """
    def __init__(self, rag_eval, graph_eval, graphrag_eval, llm_judge):
        self.rag_eval = rag_eval
        self.graph_eval = graph_eval
        self.graphrag_eval = graphrag_eval
        self.llm_judge = llm_judge
    
    def evaluate_query(self, query):
        """
        Evaluate query with all three approaches
        """
        results = {
            'query': query,
            'rag': self.rag_eval.evaluate_query(query),
            'graph_llm': self.graph_eval.evaluate_query(query),
            'graphrag_gnn': self.graphrag_eval.evaluate_query(query)
        }
        
        # Score all three with LLM judge
        results['rag_score'] = self.llm_judge.evaluate_response(
            query,
            results['rag']['response']
        )
        
        results['graph_llm_score'] = self.llm_judge.evaluate_response(
            query,
            results['graph_llm']['response']
        )
        
        results['graphrag_gnn_score'] = self.llm_judge.evaluate_response(
            query,
            results['graphrag_gnn']['response']
        )
        
        # Determine winner
        scores = {
            'RAG': results['rag_score']['overall'],
            'Graph+LLM': results['graph_llm_score']['overall'],
            'GraphRAG+GNN': results['graphrag_gnn_score']['overall']
        }
        
        results['winner'] = max(scores, key=scores.get)
        results['scores'] = scores
        
        return results
    
    def generate_comparison_report(self, results_list):
        """
        Generate comprehensive comparison report
        """
        report = {
            'rag_avg': np.mean([r['rag_score']['overall'] for r in results_list]),
            'graph_llm_avg': np.mean([r['graph_llm_score']['overall'] for r in results_list]),
            'graphrag_gnn_avg': np.mean([r['graphrag_gnn_score']['overall'] for r in results_list]),
            'rag_latency': np.mean([r['rag']['latency_ms'] for r in results_list]),
            'graph_llm_latency': np.mean([r['graph_llm']['latency_ms'] for r in results_list]),
            'graphrag_gnn_latency': np.mean([r['graphrag_gnn']['latency_ms'] for r in results_list]),
        }
        
        return report
```

## Evaluation Metrics to Add

### For GraphRAG+GNN specifically:

```python
metrics = {
    # Base metrics (same as others)
    'relevance': 0-10,
    'completeness': 0-10,
    'accuracy': 0-10,
    'specificity': 0-10,
    'clarity': 0-10,
    
    # GNN-specific metrics
    'context_quality': measure_how_well_gnn_selected_entities,
    'attention_distribution': entropy_of_attention_weights,
    'graph_coverage': what_fraction_of_graph_was_leveraged,
    'semantic_coherence': how_well_selected_context_relates_to_query,
    'efficiency': tokens_used_vs_quality_gained,
}
```

## Expected Results Comparison

```
┌────────────────┬───────────┬──────────────┬───────────────┐
│ Metric         │ RAG       │ Graph+LLM    │ GraphRAG+GNN  │
├────────────────┼───────────┼──────────────┼───────────────┤
│ Avg Score      │ 7.87      │ 7.16         │ 8.5-9.0?      │
│ Latency (ms)   │ 30-37     │ 40-51        │ 60-120        │
│ Consistency    │ 21.2% std │ 28.2% std    │ 15-20%? std   │
│ Completeness   │ 7.5       │ 6.75         │ 8.0-8.5?      │
│ Specificity    │ 9.0       │ 7.75         │ 8.5-9.0?      │
│ Speed          │ ⚡⚡⚡    │ ⚡⚡        │ ⚡            │
└────────────────┴───────────┴──────────────┴───────────────┘
```

## Implementation Roadmap

### Phase 1: Setup (1-2 hours)
- [ ] Install torch, torch-geometric
- [ ] Create `mitre_gnn_processor.py`
- [ ] Define GNN architecture
- [ ] Test with sample data

### Phase 2: Integration (2-3 hours)
- [ ] Create `GraphRAGProcessor` class
- [ ] Integrate with existing evaluation framework
- [ ] Add to `TripleEvaluator`
- [ ] Update LLM judge if needed

### Phase 3: Evaluation (2-4 hours)
- [ ] Run on test queries (5-10)
- [ ] Collect metrics
- [ ] Generate comparison report
- [ ] Analyze results

### Phase 4: Reporting (1 hour)
- [ ] Add GraphRAG+GNN results to LaTeX
- [ ] Create comparison tables
- [ ] Generate visualizations

## Quick Test Script

```python
# test_graphrag_gnn.py
from mitre_gnn_processor import GraphRAGProcessor
from sentence_transformers import SentenceTransformer

# Initialize
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
processor = GraphRAGProcessor(embedding_model, device='cpu')

# Test with sample query
query = "What techniques do threat actors use for credential theft?"

# Expected: Better than RAG (9.0), possibly equal to Graph+LLM (5.0 due to parse error)
# Could be 8.5-9.0 due to intelligent context selection
```

## Advantages of GraphRAG+GNN Over Your Current Approaches

| Aspect | RAG | Graph+LLM | GraphRAG+GNN |
|--------|-----|-----------|--------------|
| Learns from data | ❌ | ❌ | ✅ |
| Optimal context selection | ❌ | Semi (rules) | ✅ (learned) |
| Scales to large graphs | ✅ | ⚠️ Slow | ✅ |
| Training required | ❌ | ❌ | ✅ |
| Interpretability | ✅ High | ✅ High | ⚠️ Medium |
| State-of-the-art | ❌ | ⚠️ | ✅ |

## Alternative: Use Microsoft's GraphRAG Package

If you want easier implementation, Microsoft released GraphRAG:

```bash
pip install graphrag

# Uses their pre-built implementation:
# - Community detection
# - Graph clustering
# - Hierarchical indexing
# - More production-ready
```

However, building your own gives you:
- Full control over GNN architecture
- Custom integration with your framework
- Learning experience
- Ability to experiment with different GNN types (GCN, GAT, GraphSAGE)

## Questions to Consider

1. **Do you have labeled training data?**
   - Yes → Train GNN on query-response pairs
   - No → Use unsupervised GNN (node importance learning)

2. **GPU available?**
   - Yes → Can handle larger graphs, faster training
   - No → Works on CPU, just slower (30-60s added)

3. **What's your goal?**
   - Research: Build custom GNN
   - Production: Use Microsoft's GraphRAG
   - Comparison: Both approaches

## Next Steps

1. **Install dependencies**: `pip install torch torch-geometric`
2. **Create GNN module**: Copy the code above
3. **Test on sample data**: Verify it works
4. **Run evaluation**: Compare all three approaches
5. **Update LaTeX report**: Add results

Would you like me to:
1. Create the complete `mitre_gnn_processor.py` file?
2. Implement the `TripleEvaluator` class?
3. Create a test script to run the comparison?
4. Add results section to your LaTeX document?
