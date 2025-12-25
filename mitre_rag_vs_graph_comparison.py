"""
MITRE2KG: RAG vs Graph+LLM Comparison Framework
Evaluates response quality and metrics for cybersecurity queries
"""

import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import statistics
from sentence_transformers import SentenceTransformer
from arango import ArangoClient
import requests

# ============================================================================
# Configuration
# ============================================================================

ARANGODB_URL = "http://localhost:8529"
ARANGODB_USER = "root"
ARANGODB_PASS = "openSesame"
DB_NAME = "MITRE2kg"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation"""
    query: str
    approach: str  # "rag" or "graph"
    response: str
    
    # Execution metrics
    latency_ms: float
    context_size: int
    entities_retrieved: int
    relationships_retrieved: int
    
    # Response quality (manual/automated scoring 0-10)
    relevance_score: float  # How relevant is response to query?
    completeness_score: float  # Does it cover related aspects?
    accuracy_score: float  # Is info correct per MITRE?
    
    def __post_init__(self):
        self.efficiency = self.context_size / max(self.latency_ms, 1)  # entities per ms
        self.total_score = statistics.mean([
            self.relevance_score,
            self.completeness_score,
            self.accuracy_score
        ])

# ============================================================================
# Data Retrieval: Pure RAG Approach
# ============================================================================

class PureRAGApproach:
    """Semantic search only - no graph traversal"""
    
    def __init__(self, db_client, embedding_model):
        self.db = db_client
        self.model = embedding_model
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict], float]:
        """Retrieve entities via semantic search only"""
        start = time.time()
        
        # Encode query
        query_embedding = self.model.encode(query)
        
        # AQL to find most similar entities
        aql = f"""
        LET query_vec = {json.dumps(query_embedding.tolist())}
        FOR e IN entity_embeddings
            LET similarity = COSINE_SIMILARITY(query_vec, e.embedding)
            SORT similarity DESC
            LIMIT {top_k}
            RETURN {{
                entity_id: e.entity_id,
                entity_type: e.entity_type,
                entity_name: e.entity_name,
                similarity: similarity
            }}
        """
        
        results = list(self.db.aql.execute(aql))
        latency = (time.time() - start) * 1000  # Convert to ms
        
        return results, latency
    
    def generate_context(self, entities: List[Dict]) -> str:
        """Convert retrieved entities to context string for LLM"""
        context = "## Retrieved Attack Patterns and Entities:\n\n"
        for i, entity in enumerate(entities, 1):
            context += f"{i}. **{entity['entity_name']}** ({entity['entity_type']})\n"
            context += f"   - Similarity: {entity['similarity']:.3f}\n"
        return context

# ============================================================================
# Data Retrieval: Graph+LLM Approach
# ============================================================================

class GraphSearchApproach:
    """Semantic search + graph traversal for rich context"""
    
    def __init__(self, db_client, embedding_model):
        self.db = db_client
        self.model = embedding_model
    
    def search(self, query: str, top_k: int = 5, max_depth: int = 2) -> Tuple[List[Dict], List[Dict], float]:
        """Retrieve entities via semantic search + graph expansion"""
        start = time.time()
        
        # Step 1: Semantic search
        query_embedding = self.model.encode(query)
        
        aql_search = f"""
        LET query_vec = {json.dumps(query_embedding.tolist())}
        FOR e IN entity_embeddings
            LET similarity = COSINE_SIMILARITY(query_vec, e.embedding)
            SORT similarity DESC
            LIMIT {top_k}
            RETURN {{
                entity_id: e.entity_id,
                entity_type: e.entity_type,
                entity_name: e.entity_name,
                similarity: similarity
            }}
        """
        
        seed_entities = list(self.db.aql.execute(aql_search))
        entity_ids = [e['entity_id'] for e in seed_entities]
        
        # Step 2: Graph traversal - get connected entities
        aql_traverse = f"""
        LET seed_ids = {json.dumps(entity_ids)}
        FOR entity_id IN seed_ids
            FOR v, e, p IN 1..{max_depth} OUTBOUND entity_id relationships
                RETURN DISTINCT {{
                    entity_id: v._key,
                    entity_type: v.type,
                    entity_name: v.name,
                    relationship_type: e.relationship_type,
                    path_length: LENGTH(p.vertices) - 1
                }}
        """
        
        connected_entities = list(self.db.aql.execute(aql_traverse))
        latency = (time.time() - start) * 1000
        
        return seed_entities, connected_entities, latency
    
    def generate_context(self, seed_entities: List[Dict], connected: List[Dict]) -> str:
        """Convert entities and relationships to rich context"""
        context = "## Primary Attack Patterns (Semantic Match):\n\n"
        for i, entity in enumerate(seed_entities, 1):
            context += f"{i}. **{entity['entity_name']}** ({entity['entity_type']})\n"
            context += f"   - Similarity: {entity['similarity']:.3f}\n"
        
        context += "\n## Related Techniques & Threat Actors (Graph Expansion):\n\n"
        for i, entity in enumerate(connected, 1):
            context += f"{i}. **{entity['entity_name']}** ({entity['entity_type']})\n"
            context += f"   - Related via: {entity['relationship_type']}\n"
            context += f"   - Hops from seed: {entity['path_length']}\n"
        
        return context

# ============================================================================
# LLM Response Generation
# ============================================================================

class LLMResponseGenerator:
    """Wrapper for Ollama LLM"""
    
    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
    
    def generate(self, query: str, context: str, max_tokens: int = 500) -> Tuple[str, int]:
        """Generate LLM response with context"""
        
        prompt = f"""You are a cybersecurity expert specializing in MITRE ATT&CK framework.

User Query: {query}

Context from Knowledge Graph:
{context}

Based on the provided context, answer the user's query comprehensively. Reference specific techniques and threat actors from the context. If information is limited, acknowledge it.

Response:"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "num_predict": max_tokens,
                    "temperature": 0.3  # Lower temp for more consistent results
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', ''), result.get('eval_count', 0)
            else:
                return f"Error: {response.status_code}", 0
        except Exception as e:
            return f"Error generating response: {str(e)}", 0

# ============================================================================
# Evaluation Framework
# ============================================================================

class ComparisonEvaluator:
    """Main evaluation framework"""
    
    def __init__(self):
        # Connect to database
        client = ArangoClient(hosts=ARANGODB_URL)
        self.db = client.db(DB_NAME, username=ARANGODB_USER, password=ARANGODB_PASS)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize approaches
        self.rag_approach = PureRAGApproach(self.db, self.embedding_model)
        self.graph_approach = GraphSearchApproach(self.db, self.embedding_model)
        
        # Initialize LLM
        self.llm = LLMResponseGenerator()
        
        self.results: List[QueryMetrics] = []
    
    def evaluate_query(self, query: str, manual_scores: Dict[str, Dict[str, float]] = None):
        """Run both approaches and compare"""
        
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        # ---- PURE RAG APPROACH ----
        print("\n[1] Pure RAG Approach (Semantic Search Only)")
        rag_entities, rag_latency = self.rag_approach.search(query)
        rag_context = self.rag_approach.generate_context(rag_entities)
        rag_response, rag_tokens = self.llm.generate(query, rag_context)
        
        print(f"  - Retrieved: {len(rag_entities)} entities")
        print(f"  - Latency: {rag_latency:.2f} ms")
        print(f"  - Context size: {len(rag_context)} chars")
        print(f"  - Response tokens: {rag_tokens}")
        print(f"\n  Response preview: {rag_response[:200]}...")
        
        # ---- GRAPH+LLM APPROACH ----
        print("\n[2] Graph+LLM Approach (Semantic + Traversal)")
        graph_seed, graph_connected, graph_latency = self.graph_approach.search(query)
        graph_context = self.graph_approach.generate_context(graph_seed, graph_connected)
        graph_response, graph_tokens = self.llm.generate(query, graph_context)
        
        print(f"  - Seed entities: {len(graph_seed)}")
        print(f"  - Connected entities: {len(graph_connected)}")
        print(f"  - Total context: {len(graph_seed) + len(graph_connected)} entities")
        print(f"  - Latency: {graph_latency:.2f} ms")
        print(f"  - Context size: {len(graph_context)} chars")
        print(f"  - Response tokens: {graph_tokens}")
        print(f"\n  Response preview: {graph_response[:200]}...")
        
        # ---- SCORING ----
        print("\n[3] Manual Evaluation Scores (0-10)")
        
        scores = manual_scores or {}
        
        rag_metrics = QueryMetrics(
            query=query,
            approach="rag",
            response=rag_response,
            latency_ms=rag_latency,
            context_size=len(rag_entities),
            entities_retrieved=len(rag_entities),
            relationships_retrieved=0,
            relevance_score=scores.get('rag', {}).get('relevance', 0),
            completeness_score=scores.get('rag', {}).get('completeness', 0),
            accuracy_score=scores.get('rag', {}).get('accuracy', 0)
        )
        
        graph_metrics = QueryMetrics(
            query=query,
            approach="graph",
            response=graph_response,
            latency_ms=graph_latency,
            context_size=len(graph_seed) + len(graph_connected),
            entities_retrieved=len(graph_seed),
            relationships_retrieved=len(graph_connected),
            relevance_score=scores.get('graph', {}).get('relevance', 0),
            completeness_score=scores.get('graph', {}).get('completeness', 0),
            accuracy_score=scores.get('graph', {}).get('accuracy', 0)
        )
        
        self.results.append(rag_metrics)
        self.results.append(graph_metrics)
        
        print(f"\n  RAG Approach Scores:")
        print(f"    - Relevance: {rag_metrics.relevance_score:.1f}/10")
        print(f"    - Completeness: {rag_metrics.completeness_score:.1f}/10")
        print(f"    - Accuracy: {rag_metrics.accuracy_score:.1f}/10")
        print(f"    - Overall: {rag_metrics.total_score:.1f}/10")
        
        print(f"\n  Graph+LLM Approach Scores:")
        print(f"    - Relevance: {graph_metrics.relevance_score:.1f}/10")
        print(f"    - Completeness: {graph_metrics.completeness_score:.1f}/10")
        print(f"    - Accuracy: {graph_metrics.accuracy_score:.1f}/10")
        print(f"    - Overall: {graph_metrics.total_score:.1f}/10")
        
        # Comparison
        improvement = graph_metrics.total_score - rag_metrics.total_score
        efficiency_gain = (graph_metrics.efficiency - rag_metrics.efficiency) / rag_metrics.efficiency * 100
        
        print(f"\n[4] Comparison Results")
        print(f"  - Quality improvement (Graph vs RAG): {improvement:+.1f} points ({improvement/rag_metrics.total_score*100:+.1f}%)")
        print(f"  - Latency difference: {graph_latency - rag_latency:+.2f} ms ({(graph_latency-rag_latency)/rag_latency*100:+.1f}%)")
        print(f"  - Context expansion: {(rag_metrics.context_size - graph_metrics.context_size) / rag_metrics.context_size * 100:.1f}%")
        
        return rag_metrics, graph_metrics
    
    def generate_report(self) -> str:
        """Generate comprehensive comparison report"""
        
        if not self.results:
            return "No results to report"
        
        # Group by approach
        rag_results = [r for r in self.results if r.approach == "rag"]
        graph_results = [r for r in self.results if r.approach == "graph"]
        
        # Calculate aggregates
        rag_avg_score = statistics.mean([r.total_score for r in rag_results]) if rag_results else 0
        graph_avg_score = statistics.mean([r.total_score for r in graph_results]) if graph_results else 0
        
        rag_avg_latency = statistics.mean([r.latency_ms for r in rag_results]) if rag_results else 0
        graph_avg_latency = statistics.mean([r.latency_ms for r in graph_results]) if graph_results else 0
        
        rag_avg_context = statistics.mean([r.context_size for r in rag_results]) if rag_results else 0
        graph_avg_context = statistics.mean([r.context_size for r in graph_results]) if graph_results else 0
        
        report = f"""
{'='*70}
MITRE2KG: RAG vs Graph+LLM Comparison Report
{'='*70}

SUMMARY STATISTICS
{'-'*70}
Metric                          RAG             Graph+LLM       Difference
{'-'*70}
Average Quality Score          {rag_avg_score:7.2f}/10      {graph_avg_score:7.2f}/10       {graph_avg_score-rag_avg_score:+7.2f}
Average Latency (ms)           {rag_avg_latency:8.2f}       {graph_avg_latency:8.2f}       {graph_avg_latency-rag_avg_latency:+8.2f}
Average Context Size (ents)    {rag_avg_context:8.1f}       {graph_avg_context:8.1f}       {graph_avg_context-rag_avg_context:+8.1f}

QUALITY BREAKDOWN
{'-'*70}
Relevance:   RAG {statistics.mean([r.relevance_score for r in rag_results]) if rag_results else 0:.2f}  vs  Graph {statistics.mean([r.relevance_score for r in graph_results]) if graph_results else 0:.2f}
Completeness: RAG {statistics.mean([r.completeness_score for r in rag_results]) if rag_results else 0:.2f}  vs  Graph {statistics.mean([r.completeness_score for r in graph_results]) if graph_results else 0:.2f}
Accuracy:    RAG {statistics.mean([r.accuracy_score for r in rag_results]) if rag_results else 0:.2f}  vs  Graph {statistics.mean([r.accuracy_score for r in graph_results]) if graph_results else 0:.2f}

KEY FINDINGS
{'-'*70}
1. Quality Winner: {'Graph+LLM' if graph_avg_score > rag_avg_score else 'RAG'} approach
   - Score differential: {abs(graph_avg_score - rag_avg_score):.2f} points
   - Improvement: {abs(graph_avg_score - rag_avg_score)/rag_avg_score*100:.1f}%

2. Speed Winner: {'RAG' if rag_avg_latency < graph_avg_latency else 'Graph+LLM'} approach
   - Speed differential: {abs(graph_avg_latency - rag_avg_latency):.2f} ms
   - Slower by: {abs(graph_avg_latency - rag_avg_latency)/min(rag_avg_latency, graph_avg_latency)*100:.1f}%

3. Context Richness: 
   - Graph approach uses {graph_avg_context/rag_avg_context:.1f}x more entities
   - This provides {(graph_avg_context-rag_avg_context)/rag_avg_context*100:.1f}% broader context

RECOMMENDATION
{'-'*70}
{"Graph+LLM is recommended for higher quality responses with better context coverage." if graph_avg_score > rag_avg_score else "RAG is recommended for faster, more direct responses."}
Trade-off: {abs(graph_avg_latency - rag_avg_latency):.1f}ms latency for {abs(graph_avg_score - rag_avg_score):.2f} quality points.
{'='*70}
"""
        return report

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    
    evaluator = ComparisonEvaluator()
    
    # Test queries
    test_queries = [
        "What techniques do threat actors use for credential theft?",
        "How can we detect lateral movement in a network?",
        "What are the common persistence mechanisms used by APT groups?",
    ]
    
    # For demo, we'll run one query with placeholder scores
    # In practice, you'd manually score these based on response quality
    
    for query in test_queries[:1]:  # Run first query as demo
        placeholder_scores = {
            'rag': {
                'relevance': 7.5,
                'completeness': 6.5,
                'accuracy': 8.0
            },
            'graph': {
                'relevance': 8.5,
                'completeness': 8.5,
                'accuracy': 8.5
            }
        }
        
        evaluator.evaluate_query(query, placeholder_scores)
    
    # Generate report
    print(evaluator.generate_report())
    
    # Save detailed results to JSON
    with open('/home/vasanthiyer-gpu/comparison_results.json', 'w') as f:
        json.dump(
            [asdict(r) for r in evaluator.results],
            f,
            indent=2
        )
    
    print("\nDetailed results saved to: comparison_results.json")
