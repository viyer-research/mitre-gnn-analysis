"""
Triple Comparison Evaluator: RAG vs Graph+LLM vs GraphRAG+GNN
Comprehensive evaluation framework comparing three approaches
"""

import time
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from evaluating a query with one approach."""
    approach: str
    query: str
    response: str
    relevance: float
    completeness: float
    accuracy: float
    specificity: float
    clarity: float
    confidence: float
    latency_ms: float
    tokens: int
    reasoning: str = ""
    strengths: List[str] = None
    weaknesses: List[str] = None
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
        if self.weaknesses is None:
            self.weaknesses = []
    
    @property
    def overall_score(self) -> float:
        """Calculate overall score from dimensions."""
        return (self.relevance + self.completeness + self.accuracy + 
                self.specificity + self.clarity) / 5


@dataclass
class ComparisonResult:
    """Result comparing all three approaches on a query."""
    query: str
    rag_result: EvaluationResult
    graph_llm_result: EvaluationResult
    graphrag_gnn_result: EvaluationResult
    winner: str
    winner_reason: str
    
    @property
    def scores(self) -> Dict[str, float]:
        """Get scores for all approaches."""
        return {
            'RAG': self.rag_result.overall_score,
            'Graph+LLM': self.graph_llm_result.overall_score,
            'GraphRAG+GNN': self.graphrag_gnn_result.overall_score,
        }
    
    @property
    def latencies(self) -> Dict[str, float]:
        """Get latencies for all approaches."""
        return {
            'RAG': self.rag_result.latency_ms,
            'Graph+LLM': self.graph_llm_result.latency_ms,
            'GraphRAG+GNN': self.graphrag_gnn_result.latency_ms,
        }


class TripleEvaluator:
    """
    Evaluates all three approaches: RAG, Graph+LLM, and GraphRAG+GNN
    """
    
    def __init__(self, rag_evaluator, graph_llm_evaluator, graphrag_gnn_evaluator, llm_judge):
        """
        Initialize triple evaluator.
        
        Args:
            rag_evaluator: RAG evaluation instance
            graph_llm_evaluator: Graph+LLM evaluation instance
            graphrag_gnn_evaluator: GraphRAG+GNN evaluation instance
            llm_judge: LLM Judge for scoring
        """
        self.rag = rag_evaluator
        self.graph_llm = graph_llm_evaluator
        self.graphrag_gnn = graphrag_gnn_evaluator
        self.judge = llm_judge
    
    def evaluate_query(self, query: str) -> ComparisonResult:
        """
        Evaluate a query with all three approaches.
        
        Args:
            query: Query string
            
        Returns:
            ComparisonResult with all three approaches evaluated
        """
        logger.info(f"Evaluating query: {query}")
        
        # Evaluate with all three approaches
        rag_result = self._evaluate_rag(query)
        graph_llm_result = self._evaluate_graph_llm(query)
        graphrag_gnn_result = self._evaluate_graphrag_gnn(query)
        
        # Score with LLM judge
        rag_scored = self.judge.evaluate_response(query, rag_result['response'])
        graph_llm_scored = self.judge.evaluate_response(query, graph_llm_result['response'])
        graphrag_gnn_scored = self.judge.evaluate_response(query, graphrag_gnn_result['response'])
        
        # Create evaluation results
        rag_eval = EvaluationResult(
            approach='RAG',
            query=query,
            response=rag_result['response'],
            relevance=rag_scored.get('relevance', 7),
            completeness=rag_scored.get('completeness', 7),
            accuracy=rag_scored.get('accuracy', 7),
            specificity=rag_scored.get('specificity', 7),
            clarity=rag_scored.get('clarity', 7),
            confidence=rag_scored.get('confidence', 0.8),
            latency_ms=rag_result.get('latency_ms', 0),
            tokens=rag_result.get('tokens', 0),
            reasoning=rag_scored.get('reasoning', ''),
            strengths=rag_scored.get('strengths', []),
            weaknesses=rag_scored.get('weaknesses', []),
        )
        
        graph_llm_eval = EvaluationResult(
            approach='Graph+LLM',
            query=query,
            response=graph_llm_result['response'],
            relevance=graph_llm_scored.get('relevance', 7),
            completeness=graph_llm_scored.get('completeness', 7),
            accuracy=graph_llm_scored.get('accuracy', 7),
            specificity=graph_llm_scored.get('specificity', 7),
            clarity=graph_llm_scored.get('clarity', 7),
            confidence=graph_llm_scored.get('confidence', 0.8),
            latency_ms=graph_llm_result.get('latency_ms', 0),
            tokens=graph_llm_result.get('tokens', 0),
            reasoning=graph_llm_scored.get('reasoning', ''),
            strengths=graph_llm_scored.get('strengths', []),
            weaknesses=graph_llm_scored.get('weaknesses', []),
        )
        
        graphrag_gnn_eval = EvaluationResult(
            approach='GraphRAG+GNN',
            query=query,
            response=graphrag_gnn_result['response'],
            relevance=graphrag_gnn_scored.get('relevance', 7),
            completeness=graphrag_gnn_scored.get('completeness', 7),
            accuracy=graphrag_gnn_scored.get('accuracy', 7),
            specificity=graphrag_gnn_scored.get('specificity', 7),
            clarity=graphrag_gnn_scored.get('clarity', 7),
            confidence=graphrag_gnn_scored.get('confidence', 0.8),
            latency_ms=graphrag_gnn_result.get('latency_ms', 0),
            tokens=graphrag_gnn_result.get('tokens', 0),
            reasoning=graphrag_gnn_scored.get('reasoning', ''),
            strengths=graphrag_gnn_scored.get('strengths', []),
            weaknesses=graphrag_gnn_scored.get('weaknesses', []),
        )
        
        # Determine winner
        scores = {
            'RAG': rag_eval.overall_score,
            'Graph+LLM': graph_llm_eval.overall_score,
            'GraphRAG+GNN': graphrag_gnn_eval.overall_score,
        }
        
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]
        runner_up = min(scores, key=scores.get)
        
        winner_reason = f"{winner} wins with score {winner_score:.2f} (vs {runner_up}: {scores[runner_up]:.2f})"
        
        return ComparisonResult(
            query=query,
            rag_result=rag_eval,
            graph_llm_result=graph_llm_eval,
            graphrag_gnn_result=graphrag_gnn_eval,
            winner=winner,
            winner_reason=winner_reason,
        )
    
    def _evaluate_rag(self, query: str) -> Dict:
        """Evaluate RAG approach."""
        logger.info("Evaluating RAG approach...")
        try:
            result = self.rag.evaluate_single_query(query)
            return result
        except Exception as e:
            logger.error(f"RAG evaluation error: {e}")
            return {'response': f"Error: {e}", 'latency_ms': 0, 'tokens': 0}
    
    def _evaluate_graph_llm(self, query: str) -> Dict:
        """Evaluate Graph+LLM approach."""
        logger.info("Evaluating Graph+LLM approach...")
        try:
            result = self.graph_llm.evaluate_single_query(query)
            return result
        except Exception as e:
            logger.error(f"Graph+LLM evaluation error: {e}")
            return {'response': f"Error: {e}", 'latency_ms': 0, 'tokens': 0}
    
    def _evaluate_graphrag_gnn(self, query: str) -> Dict:
        """Evaluate GraphRAG+GNN approach."""
        logger.info("Evaluating GraphRAG+GNN approach...")
        try:
            result = self.graphrag_gnn.evaluate_query(query)
            return result
        except Exception as e:
            logger.error(f"GraphRAG+GNN evaluation error: {e}")
            return {'response': f"Error: {e}", 'latency_ms': 0, 'tokens': 0}
    
    def evaluate_batch(self, queries: List[str]) -> List[ComparisonResult]:
        """
        Evaluate multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of ComparisonResults
        """
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Evaluating query {i+1}/{len(queries)}")
            result = self.evaluate_query(query)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[ComparisonResult]) -> Dict:
        """
        Generate comprehensive comparison report.
        
        Args:
            results: List of ComparisonResults
            
        Returns:
            Report dictionary with summary statistics
        """
        logger.info("Generating comparison report...")
        
        rag_scores = [r.rag_result.overall_score for r in results]
        graph_llm_scores = [r.graph_llm_result.overall_score for r in results]
        graphrag_gnn_scores = [r.graphrag_gnn_result.overall_score for r in results]
        
        rag_latencies = [r.rag_result.latency_ms for r in results]
        graph_llm_latencies = [r.graph_llm_result.latency_ms for r in results]
        graphrag_gnn_latencies = [r.graphrag_gnn_result.latency_ms for r in results]
        
        winners = [r.winner for r in results]
        
        report = {
            'total_queries': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'rag': {
                'avg_score': float(np.mean(rag_scores)),
                'std_score': float(np.std(rag_scores)),
                'avg_latency_ms': float(np.mean(rag_latencies)),
                'std_latency_ms': float(np.std(rag_latencies)),
                'wins': winners.count('RAG'),
            },
            'graph_llm': {
                'avg_score': float(np.mean(graph_llm_scores)),
                'std_score': float(np.std(graph_llm_scores)),
                'avg_latency_ms': float(np.mean(graph_llm_latencies)),
                'std_latency_ms': float(np.std(graph_llm_latencies)),
                'wins': winners.count('Graph+LLM'),
            },
            'graphrag_gnn': {
                'avg_score': float(np.mean(graphrag_gnn_scores)),
                'std_score': float(np.std(graphrag_gnn_scores)),
                'avg_latency_ms': float(np.mean(graphrag_gnn_latencies)),
                'std_latency_ms': float(np.std(graphrag_gnn_latencies)),
                'wins': winners.count('GraphRAG+GNN'),
            },
            'individual_results': [
                {
                    'query': r.query,
                    'winner': r.winner,
                    'scores': r.scores,
                    'latencies': r.latencies,
                }
                for r in results
            ]
        }
        
        return report
    
    def save_results(self, results: List[ComparisonResult], filepath: str):
        """Save results to JSON file."""
        logger.info(f"Saving results to {filepath}")
        
        data = {
            'results': [
                {
                    'query': r.query,
                    'winner': r.winner,
                    'winner_reason': r.winner_reason,
                    'rag': asdict(r.rag_result),
                    'graph_llm': asdict(r.graph_llm_result),
                    'graphrag_gnn': asdict(r.graphrag_gnn_result),
                }
                for r in results
            ],
            'report': self.generate_report(results),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")


def print_comparison_table(results: List[ComparisonResult]):
    """Print a nice comparison table."""
    print("\n" + "="*100)
    print(f"{'QUERY':<50} | {'RAG':<12} | {'Graph+LLM':<12} | {'GraphRAG+GNN':<12} | {'Winner':<15}")
    print("="*100)
    
    for result in results:
        query_short = result.query[:47] + "..." if len(result.query) > 50 else result.query
        rag_score = f"{result.rag_result.overall_score:.2f}"
        graph_score = f"{result.graph_llm_result.overall_score:.2f}"
        gnn_score = f"{result.graphrag_gnn_result.overall_score:.2f}"
        
        print(f"{query_short:<50} | {rag_score:<12} | {graph_score:<12} | {gnn_score:<12} | {result.winner:<15}")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    # Example usage (requires all evaluators to be initialized)
    print("Triple Evaluator module loaded successfully")
    print("Usage: Initialize with RAG, Graph+LLM, GraphRAG+GNN evaluators and LLM Judge")
