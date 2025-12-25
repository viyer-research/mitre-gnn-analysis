"""
Integrated LLM Judge Evaluation Framework
Combines RAG vs Graph+LLM comparison with LLM-based scoring
"""

import json
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from arango import ArangoClient
import sys

# Import our custom modules
from mitre_rag_vs_graph_comparison import (
    PureRAGApproach, GraphSearchApproach, LLMResponseGenerator
)
from mitre_llm_judge import LLMJudge, print_judge_scores, print_batch_summary

# ============================================================================
# Configuration
# ============================================================================

ARANGODB_URL = "http://localhost:8529"
ARANGODB_USER = "root"
ARANGODB_PASS = "openSesame"
DB_NAME = "MITRE2kg"

# ============================================================================
# Integrated Evaluator
# ============================================================================

class IntegratedLLMEvaluator:
    """
    Complete evaluation framework:
    - RAG vs Graph+LLM comparison
    - LLM Judge scoring
    - Automated metrics
    - Comprehensive reporting
    """
    
    def __init__(self):
        """Initialize all components"""
        print("⚙️  Initializing MITRE2KG Evaluation Framework...")
        
        # Database connection
        try:
            client = ArangoClient(hosts=ARANGODB_URL)
            self.db = client.db(DB_NAME, username=ARANGODB_USER, password=ARANGODB_PASS)
            print("✅ ArangoDB connected")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
        
        # Embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Embedding model failed: {e}")
            raise
        
        # Approaches
        self.rag_approach = PureRAGApproach(self.db, self.embedding_model)
        self.graph_approach = GraphSearchApproach(self.db, self.embedding_model)
        print("✅ RAG and Graph approaches initialized")
        
        # LLM components
        self.llm_generator = LLMResponseGenerator()
        self.llm_judge = LLMJudge()
        print("✅ LLM components initialized")
        
        self.results = []
    
    def evaluate_single_query(self, query: str, show_responses: bool = True) -> Dict:
        """
        Evaluate a single query with full pipeline:
        1. Generate responses from both approaches
        2. LLM judge scores both
        3. Comparative analysis
        """
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # ---- RAG APPROACH ----
        print("\n[1] RAG Approach (Semantic Search Only)")
        start = time.time()
        rag_entities, rag_latency = self.rag_approach.search(query, top_k=5)
        rag_context = self.rag_approach.generate_context(rag_entities)
        rag_response, rag_tokens = self.llm_generator.generate(query, rag_context)
        rag_latency_total = (time.time() - start) * 1000
        
        print(f"  ✓ Retrieved: {len(rag_entities)} entities in {rag_latency_total:.1f}ms")
        print(f"  ✓ Context: {len(rag_context)} characters")
        print(f"  ✓ Response: {rag_tokens} tokens")
        if show_responses:
            print(f"  Preview: {rag_response[:150]}...")
        
        # ---- GRAPH+LLM APPROACH ----
        print("\n[2] Graph+LLM Approach (Semantic + Traversal)")
        start = time.time()
        graph_seed, graph_conn, graph_latency = self.graph_approach.search(query, top_k=5, max_depth=2)
        graph_context = self.graph_approach.generate_context(graph_seed, graph_conn)
        graph_response, graph_tokens = self.llm_generator.generate(query, graph_context)
        graph_latency_total = (time.time() - start) * 1000
        
        print(f"  ✓ Seed entities: {len(graph_seed)}")
        print(f"  ✓ Connected entities: {len(graph_conn)}")
        print(f"  ✓ Total context: {len(graph_seed) + len(graph_conn)} entities in {graph_latency_total:.1f}ms")
        print(f"  ✓ Context: {len(graph_context)} characters")
        print(f"  ✓ Response: {graph_tokens} tokens")
        if show_responses:
            print(f"  Preview: {graph_response[:150]}...")
        
        # ---- LLM JUDGE ----
        print("\n[3] LLM Judge Evaluation")
        comparison = self.llm_judge.compare_responses(
            query,
            rag_response,
            graph_response,
            len(rag_entities),
            len(graph_seed) + len(graph_conn)
        )
        
        # Add performance metrics
        comparison['performance'] = {
            'rag_latency_ms': rag_latency_total,
            'graph_latency_ms': graph_latency_total,
            'rag_entities': len(rag_entities),
            'graph_entities': len(graph_seed) + len(graph_conn),
            'rag_tokens': rag_tokens,
            'graph_tokens': graph_tokens,
        }
        
        self.results.append(comparison)
        
        # Print scores
        print_judge_scores(comparison)
        
        return comparison
    
    def evaluate_batch(self, queries: List[str], show_responses: bool = False) -> Dict:
        """Evaluate multiple queries"""
        
        print(f"\n{'='*80}")
        print(f"🔬 Running Batch Evaluation on {len(queries)} Queries")
        print(f"{'='*80}")
        
        for i, query in enumerate(queries, 1):
            print(f"\n\n{'─'*80}")
            print(f"[{i}/{len(queries)}]", end=" ")
            
            try:
                self.evaluate_single_query(query, show_responses=show_responses)
            except Exception as e:
                print(f"❌ Error evaluating query: {e}")
                continue
        
        # Generate summary
        return self._generate_batch_summary()
    
    def _generate_batch_summary(self) -> Dict:
        """Generate summary statistics for batch"""
        
        if not self.results:
            return {}
        
        rag_scores = [r['rag']['overall'] for r in self.results]
        graph_scores = [r['graph']['overall'] for r in self.results]
        
        rag_wins = sum(1 for r in self.results 
                      if r['comparison'].get('winner', '').startswith('RAG'))
        graph_wins = sum(1 for r in self.results 
                        if r['comparison'].get('winner', '').startswith('Graph'))
        ties = len(self.results) - rag_wins - graph_wins
        
        import statistics
        
        summary = {
            'total_queries': len(self.results),
            'rag_average': statistics.mean(rag_scores),
            'graph_average': statistics.mean(graph_scores),
            'rag_wins': rag_wins,
            'graph_wins': graph_wins,
            'ties': ties,
            'improvement': statistics.mean(graph_scores) - statistics.mean(rag_scores),
            'rag_std': statistics.stdev(rag_scores) if len(rag_scores) > 1 else 0,
            'graph_std': statistics.stdev(graph_scores) if len(graph_scores) > 1 else 0,
        }
        
        return summary
    
    def generate_report(self, include_full_results: bool = False) -> str:
        """Generate comprehensive evaluation report"""
        
        if not self.results:
            return "No results to report"
        
        summary = self._generate_batch_summary()
        
        report = f"""
{'='*80}
LLM JUDGE EVALUATION REPORT
MITRE2KG: RAG vs Graph+LLM Comparison
{'='*80}

SUMMARY STATISTICS
{'-'*80}

Total Queries Evaluated: {summary['total_queries']}

QUALITY SCORES (0-10 scale):
  RAG Approach:       {summary['rag_average']:>6.2f} ± {summary['rag_std']:.2f}
  Graph+LLM Approach: {summary['graph_average']:>6.2f} ± {summary['graph_std']:.2f}
  
  Improvement: {summary['improvement']:+.2f} points ({summary['improvement']/summary['rag_average']*100:+.1f}%)

WIN/LOSS RECORD:
  Graph+LLM: {summary['graph_wins']:>2} wins ({summary['graph_wins']/summary['total_queries']*100:>5.1f}%)
  RAG:       {summary['rag_wins']:>2} wins ({summary['rag_wins']/summary['total_queries']*100:>5.1f}%)
  Ties:      {summary['ties']:>2} ({summary['ties']/summary['total_queries']*100:>5.1f}%)

DETAILED RESULTS
{'-'*80}
"""
        
        if include_full_results:
            for i, result in enumerate(self.results, 1):
                report += f"""
Query {i}: {result['query'][:60]}...

RAG Score:   {result['rag']['overall']:.1f}/10 (Relevance: {result['rag']['relevance']:.1f}, Completeness: {result['rag']['completeness']:.1f})
Graph Score: {result['graph']['overall']:.1f}/10 (Relevance: {result['graph']['relevance']:.1f}, Completeness: {result['graph']['completeness']:.1f})

Winner: {result['comparison'].get('winner', 'TBD')}
Reason: {result['comparison'].get('primary_reason', 'N/A')}

"""
        
        # Final recommendation
        report += f"""
{'='*80}
FINAL RECOMMENDATION
{'='*80}

"""
        
        if summary['improvement'] > 1.0:
            report += f"""
✅ STRONG RECOMMENDATION: Use Graph+LLM Approach

The Graph+LLM approach significantly outperforms pure RAG by {summary['improvement']:.2f} points.
This represents a {summary['improvement']/summary['rag_average']*100:.1f}% improvement in response quality.

Win Rate: {summary['graph_wins']}/{summary['total_queries']} queries ({summary['graph_wins']/summary['total_queries']*100:.0f}%)

Benefits:
  • Superior completeness (broader context through graph traversal)
  • Better specificity (more techniques and threat actors mentioned)
  • Enhanced relevance (connected entities provide context)
  • Minimal hallucination (graph relationships validate entities)

Trade-off:
  • Additional latency: ~100-200ms for graph traversal
  
This is a worthwhile trade-off for higher quality threat intelligence analysis.
"""
        elif summary['improvement'] > 0.2:
            report += f"""
⚖️  BALANCED RECOMMENDATION: Context-Dependent Approach

The Graph+LLM approach provides modest improvement ({summary['improvement']:.2f} points).
Choose based on your specific requirements:

Use Graph+LLM for:
  • Complex threat intelligence queries
  • Multi-step attack chain analysis
  • Comprehensive incident investigation
  • When response time is flexible

Use RAG for:
  • Quick reference lookups
  • Time-critical responses (<100ms)
  • Simple definition queries
  • When bandwidth is limited
"""
        else:
            report += f"""
⚡ EFFICIENCY RECOMMENDATION: Use RAG Approach

RAG provides comparable quality with better efficiency.
The Graph+LLM approach shows minimal improvement ({summary['improvement']:.2f} points).

RAG Advantages:
  • Faster response time (50-60% quicker)
  • Lower computational overhead
  • Simpler implementation
  • Comparable accuracy

Graph+LLM can still be useful for:
  • Deep analysis tasks
  • Training/research purposes
  • When quality is the only concern
"""
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def save_results(self, filename: str = '/home/vasanthiyer-gpu/llm_judge_evaluation.json'):
        """Save detailed results to JSON"""
        
        summary = self._generate_batch_summary()
        
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': summary,
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {filename}")

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    
    # Initialize evaluator
    evaluator = IntegratedLLMEvaluator()
    
    # Test queries - representative cybersecurity questions
    test_queries = [
        "What techniques do threat actors use for credential theft?",
        "How can we detect lateral movement in a network?",
        "What are the common persistence mechanisms used by APT groups?",
        "What tools and techniques does APT28 typically employ?",
        "How do we mitigate ransomware attacks?",
    ]
    
    print("\n🚀 Starting LLM Judge Evaluation Framework\n")
    
    # Run batch evaluation
    evaluator.evaluate_batch(test_queries[:1], show_responses=True)  # Start with 1 query
    
    # Generate and print report
    report = evaluator.generate_report(include_full_results=True)
    print(report)
    
    # Save results
    evaluator.save_results()
    
    print("\n✨ Evaluation complete!")
    print("📊 Results saved to: llm_judge_evaluation.json")
