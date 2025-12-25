"""
Automated Metrics for RAG vs Graph+LLM Comparison
Uses BLEU, ROUGE, and semantic similarity for objective evaluation
"""

import re
import statistics
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

# ============================================================================
# Automated Metrics (No Manual Scoring Required)
# ============================================================================

class AutomatedMetrics:
    """Calculate objective metrics without manual intervention"""
    
    @staticmethod
    def extract_techniques(text: str) -> List[str]:
        """Extract MITRE ATT&CK technique IDs from text (T1234, T1234.001)"""
        pattern = r'T\d{4}(?:\.\d{3})?'
        return list(set(re.findall(pattern, text)))
    
    @staticmethod
    def extract_tactics(text: str) -> List[str]:
        """Extract tactic names from response"""
        tactics = [
            'reconnaissance', 'resource-development', 'initial-access',
            'execution', 'persistence', 'privilege-escalation', 'defense-evasion',
            'credential-access', 'discovery', 'lateral-movement', 'collection',
            'command-and-control', 'exfiltration', 'impact'
        ]
        found = []
        text_lower = text.lower()
        for tactic in tactics:
            if tactic in text_lower:
                found.append(tactic)
        return found
    
    @staticmethod
    def extract_threat_actors(text: str) -> List[str]:
        """Extract threat actor mentions (G0001, APT28, etc)"""
        # Pattern for G####
        pattern = r'G\d{4}'
        actor_ids = list(set(re.findall(pattern, text)))
        
        # Common APT names
        apt_names = ['apt1', 'apt28', 'apt29', 'lazarus', 'carbanak', 'emotet',
                     'mitre', 'cobalt', 'turla', 'winnti', 'wizard', 'wizard spider']
        actor_names = []
        text_lower = text.lower()
        for apt in apt_names:
            if apt in text_lower and apt not in ['mitre']:  # exclude false positives
                actor_names.append(apt.upper())
        
        return actor_ids + actor_names
    
    @staticmethod
    def bleu_score(reference: str, candidate: str, n: int = 2) -> float:
        """
        Calculate BLEU score (0-1) comparing reference against candidate
        Higher is better. Measures n-gram overlap.
        
        For our use case:
        - reference = expected/ideal answer
        - candidate = actual response
        """
        
        # Tokenize
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not cand_tokens:
            return 0.0
        
        # Calculate n-gram precision
        max_n = min(n, len(cand_tokens))
        precisions = []
        
        for i in range(1, max_n + 1):
            ref_ngrams = Counter(
                tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)
            )
            cand_ngrams = Counter(
                tuple(cand_tokens[j:j+i]) for j in range(len(cand_tokens) - i + 1)
            )
            
            matches = sum((cand_ngrams & ref_ngrams).values())
            possible = max(len(cand_ngrams) - i + 1, 0)
            
            if possible == 0:
                precisions.append(0.0)
            else:
                precisions.append(matches / possible)
        
        # Brevity penalty
        if len(cand_tokens) < len(ref_tokens):
            brevity_penalty = np.exp(1 - len(ref_tokens) / len(cand_tokens))
        else:
            brevity_penalty = 1.0
        
        bleu = brevity_penalty * np.exp(np.mean(np.log(precisions))) if precisions else 0.0
        return min(1.0, bleu)
    
    @staticmethod
    def rouge_l_score(reference: str, candidate: str) -> float:
        """
        Calculate ROUGE-L score (longest common subsequence)
        Measures longest matching sequence of words
        """
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Longest common subsequence
        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_len = dp[m][n]
        
        # Precision and recall
        precision = lcs_len / n if n > 0 else 0
        recall = lcs_len / m if m > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        # F-score (harmonic mean)
        f_score = 2 * precision * recall / (precision + recall)
        return min(1.0, f_score)
    
    @staticmethod
    def information_density(response: str, context_size: int) -> float:
        """
        Information Density = unique information / context size
        
        Higher is better: means more unique/novel information
        per unit of context provided
        """
        # Count unique information elements
        techniques = len(AutomatedMetrics.extract_techniques(response))
        tactics = len(AutomatedMetrics.extract_tactics(response))
        actors = len(AutomatedMetrics.extract_threat_actors(response))
        unique_words = len(set(response.lower().split()))
        
        info_count = techniques * 2 + tactics * 1.5 + actors * 1.5 + unique_words * 0.01
        
        if context_size == 0:
            return 0.0
        
        density = info_count / context_size
        return min(1.0, density)  # Cap at 1.0
    
    @staticmethod
    def coverage_score(response: str, expected_topics: List[str]) -> float:
        """
        Coverage = how many expected topics are mentioned
        
        expected_topics: list of technique IDs or keywords expected in answer
        """
        response_lower = response.lower()
        covered = sum(1 for topic in expected_topics 
                     if topic.lower() in response_lower)
        
        if not expected_topics:
            return 1.0
        
        return covered / len(expected_topics)
    
    @staticmethod
    def hallucination_score(response: str, known_entities: List[str]) -> float:
        """
        Hallucination Score = how many mentioned entities are actually valid
        
        1.0 = no hallucinations
        0.5 = half the mentioned entities don't exist
        0.0 = all mentioned entities are hallucinated
        """
        # Extract technique IDs
        mentioned_techniques = AutomatedMetrics.extract_techniques(response)
        
        if not mentioned_techniques:
            return 1.0  # No techniques mentioned, so no hallucinations
        
        valid = sum(1 for tech in mentioned_techniques if tech in known_entities)
        
        return valid / len(mentioned_techniques)
    
    @staticmethod
    def specificity_score(response: str) -> float:
        """
        Specificity Score: Does response mention specific techniques/actors?
        
        1.0 = very specific (lots of T#### and G#### mentions)
        0.5 = medium specificity
        0.0 = very generic/vague
        """
        techniques = len(AutomatedMetrics.extract_techniques(response))
        actors = len(AutomatedMetrics.extract_threat_actors(response))
        specific_elements = techniques + actors
        
        # Normalize by response length
        words = len(response.split())
        if words == 0:
            return 0.0
        
        specificity = min(1.0, specific_elements / (words / 50))  # ~1 specific per 50 words ideal
        return specificity

# ============================================================================
# Comparison Metrics Calculator
# ============================================================================

class ComparisonMetricsCalculator:
    """Calculate all metrics for RAG vs Graph comparison"""
    
    def __init__(self, known_techniques: List[str] = None):
        """
        known_techniques: list of valid MITRE ATT&CK technique IDs for 
                         hallucination detection
        """
        self.known_techniques = known_techniques or []
        self.metrics = AutomatedMetrics()
    
    def calculate_all(self, 
                     query: str,
                     rag_response: str,
                     graph_response: str,
                     expected_topics: List[str] = None,
                     rag_context_size: int = 0,
                     graph_context_size: int = 0) -> Dict:
        """
        Calculate all metrics for both approaches
        
        Returns dict with all metrics and comparison
        """
        
        expected_topics = expected_topics or []
        
        # ---- RAG METRICS ----
        rag_bleu = self.metrics.bleu_score(graph_response, rag_response)
        rag_rouge = self.metrics.rouge_l_score(graph_response, rag_response)
        rag_density = self.metrics.information_density(rag_response, rag_context_size)
        rag_coverage = self.metrics.coverage_score(rag_response, expected_topics)
        rag_hallucination = self.metrics.hallucination_score(
            rag_response, self.known_techniques
        )
        rag_specificity = self.metrics.specificity_score(rag_response)
        rag_techniques = len(self.metrics.extract_techniques(rag_response))
        rag_actors = len(self.metrics.extract_threat_actors(rag_response))
        
        # ---- GRAPH METRICS ----
        graph_bleu = self.metrics.bleu_score(rag_response, graph_response)
        graph_rouge = self.metrics.rouge_l_score(rag_response, graph_response)
        graph_density = self.metrics.information_density(graph_response, graph_context_size)
        graph_coverage = self.metrics.coverage_score(graph_response, expected_topics)
        graph_hallucination = self.metrics.hallucination_score(
            graph_response, self.known_techniques
        )
        graph_specificity = self.metrics.specificity_score(graph_response)
        graph_techniques = len(self.metrics.extract_techniques(graph_response))
        graph_actors = len(self.metrics.extract_threat_actors(graph_response))
        
        # ---- COMPOSITE SCORES ----
        # Quality: average of coverage, specificity, hallucination (inverted), and text metrics
        rag_quality = statistics.mean([
            rag_coverage,
            rag_specificity,
            rag_hallucination,
            (rag_bleu + rag_rouge) / 2
        ])
        
        graph_quality = statistics.mean([
            graph_coverage,
            graph_specificity,
            graph_hallucination,
            (graph_bleu + graph_rouge) / 2
        ])
        
        return {
            'query': query,
            'rag': {
                'bleu': rag_bleu,
                'rouge_l': rag_rouge,
                'information_density': rag_density,
                'coverage': rag_coverage,
                'hallucination': rag_hallucination,
                'specificity': rag_specificity,
                'techniques_mentioned': rag_techniques,
                'actors_mentioned': rag_actors,
                'composite_quality': rag_quality,
            },
            'graph': {
                'bleu': graph_bleu,
                'rouge_l': graph_rouge,
                'information_density': graph_density,
                'coverage': graph_coverage,
                'hallucination': graph_hallucination,
                'specificity': graph_specificity,
                'techniques_mentioned': graph_techniques,
                'actors_mentioned': graph_actors,
                'composite_quality': graph_quality,
            },
            'comparison': {
                'quality_delta': graph_quality - rag_quality,
                'quality_winner': 'graph' if graph_quality > rag_quality else 'rag' if rag_quality > graph_quality else 'tie',
                'coverage_delta': graph_coverage - rag_coverage,
                'specificity_delta': graph_specificity - rag_specificity,
                'hallucination_delta': graph_hallucination - rag_hallucination,
            }
        }
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """Pretty-print metrics comparison"""
        
        print(f"\n{'='*80}")
        print(f"Query: {metrics['query']}")
        print(f"{'='*80}")
        
        print(f"\n{'RAG':^40} | {'Graph+LLM':^40}")
        print(f"{'-'*40}-+-{'-'*40}")
        
        # BLEU Score
        rag_bleu = metrics['rag']['bleu']
        graph_bleu = metrics['graph']['bleu']
        print(f"BLEU Score:          {rag_bleu:>6.3f}        |           {graph_bleu:>6.3f}")
        
        # ROUGE-L Score
        rag_rouge = metrics['rag']['rouge_l']
        graph_rouge = metrics['graph']['rouge_l']
        print(f"ROUGE-L Score:       {rag_rouge:>6.3f}        |           {graph_rouge:>6.3f}")
        
        # Information Density
        rag_density = metrics['rag']['information_density']
        graph_density = metrics['graph']['information_density']
        print(f"Info Density:        {rag_density:>6.3f}        |           {graph_density:>6.3f}")
        
        # Coverage
        rag_coverage = metrics['rag']['coverage']
        graph_coverage = metrics['graph']['coverage']
        print(f"Topic Coverage:      {rag_coverage:>6.3f}        |           {graph_coverage:>6.3f}")
        
        # Hallucination
        rag_hall = metrics['rag']['hallucination']
        graph_hall = metrics['graph']['hallucination']
        print(f"Accuracy (no hall.): {rag_hall:>6.3f}        |           {graph_hall:>6.3f}")
        
        # Specificity
        rag_spec = metrics['rag']['specificity']
        graph_spec = metrics['graph']['specificity']
        print(f"Specificity:         {rag_spec:>6.3f}        |           {graph_spec:>6.3f}")
        
        # Entity mentions
        print(f"\nTechniques Mentioned: {metrics['rag']['techniques_mentioned']:>3}           |           {metrics['graph']['techniques_mentioned']:>3}")
        print(f"Actors Mentioned:     {metrics['rag']['actors_mentioned']:>3}           |           {metrics['graph']['actors_mentioned']:>3}")
        
        # Composite Quality
        rag_quality = metrics['rag']['composite_quality']
        graph_quality = metrics['graph']['composite_quality']
        print(f"\n{'COMPOSITE QUALITY':^40} | {'':<40}")
        print(f"{rag_quality:>6.3f} / 1.000        |           {graph_quality:>6.3f} / 1.000")
        
        # Winner
        winner = metrics['comparison']['quality_winner'].upper()
        delta = metrics['comparison']['quality_delta']
        delta_pct = (delta / rag_quality * 100) if rag_quality > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"WINNER: {winner} approach")
        print(f"Quality Improvement: {delta:+.4f} points ({delta_pct:+.1f}%)")
        print(f"{'='*80}")

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    
    # Example known techniques (in practice, load from database)
    known_techniques = [
        'T1110', 'T1187', 'T1056', 'T1040', 'T1056.004',
        'T1087', 'T1010', 'T1217', 'T1580', 'T1547', 'T1547.001'
    ]
    
    calculator = ComparisonMetricsCalculator(known_techniques)
    
    # Example responses
    rag_response = """
    For credential theft attacks, threat actors use several key techniques:
    
    T1110 - Brute Force: Attackers attempt repeated login attempts
    T1187 - Forced Authentication: Forcing users to authenticate to attacker-controlled systems
    T1056 - Input Capture: Monitoring user input to steal credentials
    T1040 - Network Sniffing: Capturing network traffic to extract credentials
    
    These techniques are commonly used by threat actors to gain initial access.
    """
    
    graph_response = """
    Credential theft in MITRE ATT&CK framework involves multiple techniques:
    
    **Primary Techniques:**
    T1110 - Brute Force: Used by G0001 (APT1) and G0007 (APT28)
    T1187 - Forced Authentication: Common in lateral movement
    T1056 - Input Capture: Captured by detection systems
    T1056.004 - Credential API Hooking: Windows-specific variant
    T1040 - Network Sniffing: Network-level attacks
    
    **Related Tactics:**
    - Credential Access (TA0006)
    - Discovery (TA0007)
    
    **Threat Actors:**
    - APT28: Uses T1110 and T1040
    - Lazarus: Uses T1056 variants
    
    **Mitigations:**
    - Multi-factor authentication
    - Network monitoring
    - EDR solutions
    
    **CISA Advisories:**
    Referenced in aa24-060a and related incidents
    """
    
    # Calculate metrics
    metrics = calculator.calculate_all(
        query="What techniques do threat actors use for credential theft?",
        rag_response=rag_response,
        graph_response=graph_response,
        expected_topics=['T1110', 'T1187', 'T1056', 'threat actors', 'APT'],
        rag_context_size=50,
        graph_context_size=150
    )
    
    # Print results
    ComparisonMetricsCalculator.print_metrics(metrics)
    
    # Show individual metric explanations
    print("\n" + "="*80)
    print("METRIC DEFINITIONS")
    print("="*80)
    print("""
BLEU Score (0-1):
  Measures n-gram overlap between responses. Higher is better.
  Range: 0 = completely different, 1 = identical

ROUGE-L Score (0-1):
  Measures longest common subsequence of words. Higher is better.
  Range: 0 = no overlap, 1 = perfect match

Information Density (0-1):
  Unique information per unit context. Higher is better.
  Measures efficiency of information delivery

Topic Coverage (0-1):
  Percentage of expected topics mentioned. Higher is better.
  1.0 = all expected topics covered

Accuracy/Hallucination (0-1):
  Percentage of mentioned entities that are valid. Higher is better.
  1.0 = no hallucinations, 0.5 = half are invalid

Specificity (0-1):
  How many specific entities (T####, G####) mentioned. Higher is better.
  More specific details = better for cybersecurity

Composite Quality (0-1):
  Average of all metrics. Higher is better.
  Best overall measure of response quality
    """)
