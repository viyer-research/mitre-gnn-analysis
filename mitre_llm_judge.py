"""
LLM Judge for RAG vs Graph+LLM Comparison
Uses Ollama/LLM to objectively evaluate and score responses
"""

import json
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import requests
import statistics

# ============================================================================
# Configuration
# ============================================================================

OLLAMA_URL = "http://localhost:11434"
JUDGE_MODEL = "llama3.1:8b"  # LLM judge model
RESPONSE_MODEL = "llama3.1:8b"  # Model generating responses

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class JudgeScores:
    """Scores from LLM judge"""
    relevance: float  # 0-10
    completeness: float  # 0-10
    accuracy: float  # 0-10
    specificity: float  # 0-10
    clarity: float  # 0-10
    confidence: float  # 0-1 (judge's confidence in scores)
    reasoning: str  # Explanation of scores
    strengths: List[str]  # What this response does well
    weaknesses: List[str]  # What could be improved
    overall_score: float  # Composite 0-10

# ============================================================================
# LLM Judge
# ============================================================================

class LLMJudge:
    """
    Uses LLM to evaluate responses on multiple dimensions
    More consistent and objective than manual scoring
    """
    
    def __init__(self, base_url: str = OLLAMA_URL, model: str = JUDGE_MODEL):
        self.base_url = base_url
        self.model = model
    
    def _call_llm(self, prompt: str, temperature: float = 0.2) -> str:
        """Call LLM with structured prompt"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "num_predict": 2000
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def evaluate_response(self, 
                         query: str,
                         response: str,
                         context_size: int = 0,
                         approach_name: str = "Unknown") -> JudgeScores:
        """
        Evaluate a single response on multiple dimensions
        
        Args:
            query: Original user query
            response: Response text to evaluate
            context_size: Number of entities in context
            approach_name: "RAG" or "Graph+LLM" for reference
        """
        
        evaluation_prompt = f"""You are an expert cybersecurity researcher specializing in MITRE ATT&CK framework evaluation.

You will evaluate a response to a user query and score it on multiple dimensions.

USER QUERY:
{query}

RESPONSE TO EVALUATE ({approach_name}):
{response}

Context Size: {context_size} entities

Please evaluate this response using the following criteria (each 0-10):

1. RELEVANCE (0-10): How directly does the response address the user's query?
   - 10: Perfect match, directly answers all aspects
   - 8-9: Directly answers with minor tangents
   - 6-7: Mostly relevant
   - 4-5: Somewhat relevant
   - 0-3: Not relevant

2. COMPLETENESS (0-10): Does it cover the full scope of the topic?
   - 10: Comprehensive coverage of techniques, tactics, actors, detection, mitigation
   - 8-9: Covers most important aspects
   - 6-7: Covers multiple aspects
   - 4-5: Covers basic information
   - 0-3: Surface-level or incomplete

3. ACCURACY (0-10): Is information factually correct per MITRE ATT&CK?
   - 10: All information accurate
   - 8-9: Mostly accurate
   - 6-7: Some minor inaccuracies
   - 4-5: Multiple inaccuracies
   - 0-3: Highly inaccurate or hallucinated

4. SPECIFICITY (0-10): Does it mention specific techniques (T####) and actors (G####)?
   - 10: Abundant specific references
   - 8-9: Good specific examples
   - 6-7: Some specifics
   - 4-5: Few specifics, mostly generic
   - 0-3: No specific references

5. CLARITY (0-10): Is the response well-structured and easy to understand?
   - 10: Excellent organization and clarity
   - 8-9: Clear with good structure
   - 6-7: Understandable but could be clearer
   - 4-5: Somewhat confusing
   - 0-3: Unclear or poorly organized

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "relevance": <0-10>,
    "completeness": <0-10>,
    "accuracy": <0-10>,
    "specificity": <0-10>,
    "clarity": <0-10>,
    "overall_score": <average of above>,
    "confidence": <0.0-1.0>,
    "reasoning": "<2-3 sentence summary of evaluation>",
    "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"]
}}

Evaluate objectively. Be critical but fair."""
        
        # Get evaluation from LLM
        response_text = self._call_llm(evaluation_prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                scores_dict = json.loads(json_str)
            else:
                # If no JSON found, return defaults
                scores_dict = {
                    "relevance": 5, "completeness": 5, "accuracy": 5,
                    "specificity": 5, "clarity": 5, "overall_score": 5,
                    "confidence": 0.3,
                    "reasoning": "Failed to parse evaluation",
                    "strengths": [], "weaknesses": []
                }
        except json.JSONDecodeError:
            scores_dict = {
                "relevance": 5, "completeness": 5, "accuracy": 5,
                "specificity": 5, "clarity": 5, "overall_score": 5,
                "confidence": 0.3,
                "reasoning": f"Parse error: {response_text[:100]}",
                "strengths": [], "weaknesses": []
            }
        
        return JudgeScores(
            relevance=scores_dict.get('relevance', 5),
            completeness=scores_dict.get('completeness', 5),
            accuracy=scores_dict.get('accuracy', 5),
            specificity=scores_dict.get('specificity', 5),
            clarity=scores_dict.get('clarity', 5),
            confidence=scores_dict.get('confidence', 0.3),
            reasoning=scores_dict.get('reasoning', ''),
            strengths=scores_dict.get('strengths', []),
            weaknesses=scores_dict.get('weaknesses', []),
            overall_score=scores_dict.get('overall_score', 5)
        )
    
    def compare_responses(self,
                         query: str,
                         rag_response: str,
                         graph_response: str,
                         rag_context_size: int = 0,
                         graph_context_size: int = 0) -> Dict[str, Any]:
        """
        Compare two responses and determine which is better
        
        Returns: Dict with scores for both + comparative analysis
        """
        
        print("🔍 Evaluating RAG approach...")
        rag_scores = self.evaluate_response(query, rag_response, rag_context_size, "RAG")
        
        print("🔍 Evaluating Graph+LLM approach...")
        graph_scores = self.evaluate_response(query, graph_response, graph_context_size, "Graph+LLM")
        
        # Comparative analysis
        print("📊 Performing comparative analysis...")
        winner = self._determine_winner(query, rag_response, graph_response, 
                                       rag_scores, graph_scores)
        
        return {
            'query': query,
            'rag': {
                'relevance': rag_scores.relevance,
                'completeness': rag_scores.completeness,
                'accuracy': rag_scores.accuracy,
                'specificity': rag_scores.specificity,
                'clarity': rag_scores.clarity,
                'overall': rag_scores.overall_score,
                'confidence': rag_scores.confidence,
                'reasoning': rag_scores.reasoning,
                'strengths': rag_scores.strengths,
                'weaknesses': rag_scores.weaknesses,
            },
            'graph': {
                'relevance': graph_scores.relevance,
                'completeness': graph_scores.completeness,
                'accuracy': graph_scores.accuracy,
                'specificity': graph_scores.specificity,
                'clarity': graph_scores.clarity,
                'overall': graph_scores.overall_score,
                'confidence': graph_scores.confidence,
                'reasoning': graph_scores.reasoning,
                'strengths': graph_scores.strengths,
                'weaknesses': graph_scores.weaknesses,
            },
            'comparison': winner
        }
    
    def _determine_winner(self, query: str, rag_resp: str, graph_resp: str,
                         rag_scores: JudgeScores, graph_scores: JudgeScores) -> Dict:
        """Use LLM to provide comparative analysis"""
        
        comparison_prompt = f"""You are an expert cybersecurity researcher evaluating two AI responses.

QUERY: {query}

RESPONSE A (RAG - Semantic Search Only):
{rag_resp[:500]}...

RESPONSE B (Graph+LLM - Semantic Search + Graph Traversal):
{graph_resp[:500]}...

SCORES FOR RESPONSE A (RAG):
- Overall: {rag_scores.overall_score:.1f}/10
- Relevance: {rag_scores.relevance:.1f}/10
- Completeness: {rag_scores.completeness:.1f}/10
- Accuracy: {rag_scores.accuracy:.1f}/10
- Specificity: {rag_scores.specificity:.1f}/10

SCORES FOR RESPONSE B (Graph+LLM):
- Overall: {graph_scores.overall_score:.1f}/10
- Relevance: {graph_scores.relevance:.1f}/10
- Completeness: {graph_scores.completeness:.1f}/10
- Accuracy: {graph_scores.accuracy:.1f}/10
- Specificity: {graph_scores.specificity:.1f}/10

Please provide a brief comparative analysis. Return as JSON:

{{
    "winner": "<RAG or Graph+LLM>",
    "primary_reason": "<main reason for winner>",
    "key_difference": "<what sets winner apart>",
    "recommendation": "<which to use and when>",
    "margin": <score difference 0-10>
}}"""
        
        response_text = self._call_llm(comparison_prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {
                    "winner": "Tie",
                    "primary_reason": "Unable to determine",
                    "key_difference": "Scores are too close",
                    "recommendation": "Use both for different scenarios",
                    "margin": 0
                }
        except:
            return {
                "winner": "Tie",
                "primary_reason": "Analysis error",
                "key_difference": "Unable to analyze",
                "recommendation": "Manual review recommended",
                "margin": 0
            }

# ============================================================================
# Batch Evaluator
# ============================================================================

class BatchLLMEvaluator:
    """Evaluate multiple queries with LLM judge"""
    
    def __init__(self, rag_approach, graph_approach, llm_generator):
        """
        Args:
            rag_approach: PureRAGApproach instance
            graph_approach: GraphSearchApproach instance
            llm_generator: LLMResponseGenerator instance
        """
        self.rag = rag_approach
        self.graph = graph_approach
        self.llm_gen = llm_generator
        self.judge = LLMJudge()
        self.results = []
    
    def evaluate_queries(self, queries: List[str]) -> Dict[str, Any]:
        """Run full evaluation on multiple queries"""
        
        print(f"\n{'='*70}")
        print(f"Running LLM Judge Evaluation on {len(queries)} Queries")
        print(f"{'='*70}\n")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing: {query[:60]}...")
            
            # Get RAG response
            rag_entities, rag_latency = self.rag.search(query)
            rag_context = self.rag.generate_context(rag_entities)
            rag_response, _ = self.llm_gen.generate(query, rag_context)
            
            # Get Graph response
            graph_seed, graph_conn, graph_latency = self.graph.search(query)
            graph_context = self.graph.generate_context(graph_seed, graph_conn)
            graph_response, _ = self.llm_gen.generate(query, graph_context)
            
            # Judge both
            comparison = self.judge.compare_responses(
                query,
                rag_response,
                graph_response,
                len(rag_entities),
                len(graph_seed) + len(graph_conn)
            )
            
            self.results.append(comparison)
            
            # Print scores
            print(f"  RAG Score:   {comparison['rag']['overall']:.1f}/10")
            print(f"  Graph Score: {comparison['graph']['overall']:.1f}/10")
            print(f"  Winner: {comparison['comparison'].get('winner', 'TBD')}")
        
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary statistics"""
        
        if not self.results:
            return {}
        
        rag_scores = [r['rag']['overall'] for r in self.results]
        graph_scores = [r['graph']['overall'] for r in self.results]
        
        rag_wins = sum(1 for r in self.results if 'RAG' in r['comparison'].get('winner', ''))
        graph_wins = sum(1 for r in self.results if 'Graph' in r['comparison'].get('winner', ''))
        ties = len(self.results) - rag_wins - graph_wins
        
        return {
            'total_queries': len(self.results),
            'rag_average': statistics.mean(rag_scores),
            'graph_average': statistics.mean(graph_scores),
            'rag_wins': rag_wins,
            'graph_wins': graph_wins,
            'ties': ties,
            'improvement': statistics.mean(graph_scores) - statistics.mean(rag_scores),
            'queries': self.results
        }

# ============================================================================
# Utility Functions
# ============================================================================

def print_judge_scores(comparison: Dict) -> None:
    """Pretty-print LLM judge evaluation"""
    
    print(f"\n{'='*80}")
    print(f"Query: {comparison['query']}")
    print(f"{'='*80}")
    
    print(f"\n{'DIMENSION':<20} {'RAG':>10} {'Graph+LLM':>15} {'Winner':>15}")
    print(f"{'-'*80}")
    
    dims = ['relevance', 'completeness', 'accuracy', 'specificity', 'clarity']
    
    for dim in dims:
        rag_val = comparison['rag'][dim]
        graph_val = comparison['graph'][dim]
        winner = '→ Graph' if graph_val > rag_val else '→ RAG' if rag_val > graph_val else '  Tie'
        print(f"{dim.title():<20} {rag_val:>10.1f} {graph_val:>15.1f} {winner:>15}")
    
    print(f"{'-'*80}")
    print(f"{'OVERALL':<20} {comparison['rag']['overall']:>10.1f} {comparison['graph']['overall']:>15.1f}")
    
    print(f"\n{'RAG APPROACH':<40} {'GRAPH+LLM APPROACH':<40}")
    print(f"{'-'*80}")
    
    print(f"Reasoning: {comparison['rag']['reasoning']}")
    print(f"Reasoning: {comparison['graph']['reasoning']}")
    
    print(f"\nStrengths:")
    print(f"  RAG: {', '.join(comparison['rag']['strengths'][:2])}")
    print(f"  Graph: {', '.join(comparison['graph']['strengths'][:2])}")
    
    print(f"\nWeaknesses:")
    print(f"  RAG: {', '.join(comparison['rag']['weaknesses'][:2])}")
    print(f"  Graph: {', '.join(comparison['graph']['weaknesses'][:2])}")
    
    print(f"\n{'='*80}")
    print(f"Comparative Analysis")
    print(f"{'='*80}")
    print(f"Winner: {comparison['comparison'].get('winner', 'TBD').upper()}")
    print(f"Reason: {comparison['comparison'].get('primary_reason', 'N/A')}")
    print(f"Key Difference: {comparison['comparison'].get('key_difference', 'N/A')}")
    print(f"Recommendation: {comparison['comparison'].get('recommendation', 'N/A')}")
    print(f"Margin: {comparison['comparison'].get('margin', 0):.1f} points")

def print_batch_summary(summary: Dict) -> None:
    """Print summary of batch evaluation"""
    
    print(f"\n{'='*80}")
    print(f"BATCH EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nQueries Evaluated: {summary['total_queries']}")
    print(f"\nAverage Scores:")
    print(f"  RAG:       {summary['rag_average']:.2f} / 10")
    print(f"  Graph+LLM: {summary['graph_average']:.2f} / 10")
    print(f"  Improvement: {summary['improvement']:+.2f} points ({summary['improvement']/summary['rag_average']*100:+.1f}%)")
    
    print(f"\nWin/Loss Record:")
    print(f"  Graph+LLM: {summary['graph_wins']} wins ({summary['graph_wins']/summary['total_queries']*100:.0f}%)")
    print(f"  RAG:       {summary['rag_wins']} wins ({summary['rag_wins']/summary['total_queries']*100:.0f}%)")
    print(f"  Ties:      {summary['ties']} ({summary['ties']/summary['total_queries']*100:.0f}%)")
    
    if summary['improvement'] > 0.5:
        print(f"\n✅ RECOMMENDATION: Use Graph+LLM approach")
        print(f"   Graph+LLM is significantly better across queries.")
    elif summary['improvement'] < -0.5:
        print(f"\n✅ RECOMMENDATION: Use RAG approach")
        print(f"   RAG is more efficient with comparable quality.")
    else:
        print(f"\n⚖️  RECOMMENDATION: Use context-dependent approach")
        print(f"   Choose based on latency vs quality trade-off for your use case.")
    
    print(f"\n{'='*80}")

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    
    # Example: Using judge to evaluate two responses
    judge = LLMJudge()
    
    example_query = "What techniques do threat actors use for credential theft?"
    
    example_rag = """
    Threat actors use several credential theft techniques:
    - Brute Force (T1110): Repeated login attempts
    - Network Sniffing (T1040): Capturing network traffic
    - Input Capture (T1056): Monitoring user input
    """
    
    example_graph = """
    Credential theft in MITRE ATT&CK involves:
    
    Primary Techniques:
    - T1110 (Brute Force): Used by APT28, APT1
    - T1040 (Network Sniffing): Common in lateral movement
    - T1056 (Input Capture): Keylogging, clipboard monitoring
    - T1187 (Forced Authentication): Active Directory attacks
    
    Related Tactics:
    - Credential Access (TA0006)
    - Persistence (TA0003)
    
    Threat Actors:
    - APT28: Uses T1110, T1040
    - Lazarus: Uses T1056 variants
    
    Mitigations:
    - Multi-factor authentication
    - Network monitoring
    - EDR solutions
    """
    
    print("🔬 LLM Judge Evaluation System")
    print("=" * 70)
    
    comparison = judge.compare_responses(
        example_query,
        example_rag,
        example_graph,
        rag_context_size=5,
        graph_context_size=20
    )
    
    print_judge_scores(comparison)
    
    # Save results
    with open('/home/vasanthiyer-gpu/llm_judge_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\n✅ Results saved to llm_judge_results.json")
