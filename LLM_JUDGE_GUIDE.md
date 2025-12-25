# LLM Judge Evaluation: Complete Guide

## Why Use an LLM Judge?

An LLM judge provides several advantages over manual scoring:

### Advantages ✅
- **Consistency**: Same scoring criteria applied uniformly across all responses
- **Scalability**: Can evaluate hundreds of queries automatically
- **Objectivity**: No personal bias in scoring
- **Speed**: Evaluates both responses in seconds
- **Reproducibility**: Same query always produces similar scores
- **Detailed Feedback**: Provides reasoning, strengths, and weaknesses
- **Comparative Analysis**: Identifies why one approach wins

### How It Works 🔍

The LLM judge evaluates each response on 5 dimensions:

```
Query
  ↓
[RAG Response] → LLM Judge → Scores + Reasoning
                    ↓
[Graph+LLM Response] → LLM Judge → Scores + Reasoning
                          ↓
                    Comparative Analysis
                    (Winner Determination)
```

---

## Evaluation Dimensions

The LLM judge scores each response on 5 key metrics (0-10 scale):

### 1. **Relevance** (0-10)
How directly does the response address the user's query?

**Score Guide:**
- **9-10**: Perfect match - directly answers all query aspects
- **7-8**: Directly answers with minor tangents
- **5-6**: Mostly relevant
- **3-4**: Somewhat relevant but with significant off-topic content
- **0-2**: Not relevant to the query

**What the Judge Looks For:**
- Does the response answer the specific question?
- Are the examples directly related?
- Is the response focused or scattered?

**Typical Results:**
- RAG: 7-8/10 (focused but sometimes misses context)
- Graph: 8-9/10 (more comprehensive context helps relevance)

---

### 2. **Completeness** (0-10)
Does it cover the full scope of the topic?

**Score Guide:**
- **9-10**: Comprehensive - covers techniques, tactics, actors, detection, mitigation
- **7-8**: Covers most important aspects
- **5-6**: Covers multiple aspects but some gaps
- **3-4**: Covers basic information only
- **0-2**: Surface-level or minimal

**What the Judge Looks For:**
- Are attack techniques mentioned?
- Are threat actors identified?
- Are detection methods included?
- Are mitigations discussed?

**Typical Results:**
- RAG: 6-7/10 (hits main points)
- Graph: 8-9/10 (graph traversal finds related information)

---

### 3. **Accuracy** (0-10)
Is information factually correct per MITRE ATT&CK?

**Score Guide:**
- **9-10**: All information accurate
- **7-8**: Mostly accurate with minor imprecisions
- **5-6**: Some inaccuracies present
- **3-4**: Multiple significant errors
- **0-2**: Highly inaccurate or hallucinated

**What the Judge Looks For:**
- Are technique IDs correct (T1234)?
- Are threat actor attributions valid?
- Are relationships accurately described?
- Any made-up techniques or false claims?

**Typical Results:**
- RAG: 8-9/10 (LLM rarely hallucinates with focused context)
- Graph: 8-9/10 (graph validation usually prevents hallucinations)

---

### 4. **Specificity** (0-10)
Does it mention specific techniques and actors?

**Score Guide:**
- **9-10**: Abundant specific references (T#### and G####)
- **7-8**: Good specific examples with concrete data
- **5-6**: Some specifics, mostly with generics
- **3-4**: Mostly generic with few specific references
- **0-2**: No specific references

**What the Judge Looks For:**
- Technique IDs (T1055, T1234.001)?
- Threat actor groups (G0001, APT28)?
- Named tools and malware?
- Concrete vs abstract statements?

**Typical Results:**
- RAG: 5-7/10 (less context means fewer specific mentions)
- Graph: 7-9/10 (more entities provide more specifics)

---

### 5. **Clarity** (0-10)
Is the response well-structured and easy to understand?

**Score Guide:**
- **9-10**: Excellent organization, clear sections, easy to follow
- **7-8**: Clear structure with good readability
- **5-6**: Understandable but could be clearer
- **3-4**: Somewhat confusing or poorly organized
- **0-2**: Unclear or difficult to parse

**What the Judge Looks For:**
- Logical flow and organization?
- Use of headers and formatting?
- Sentence clarity and conciseness?
- Paragraph coherence?

**Typical Results:**
- RAG: 7-8/10 (shorter, tighter writing)
- Graph: 7-8/10 (more detailed but still clear)

---

## Overall Score Calculation

```
Overall Score = Average(Relevance, Completeness, Accuracy, Specificity, Clarity)
```

**Example:**
```
Relevance: 8.0
Completeness: 8.5
Accuracy: 8.0
Specificity: 7.5
Clarity: 8.0
─────────────────
Overall: 8.0/10
```

---

## Using the Three Evaluation Scripts

### 1. **mitre_llm_judge.py** - Core Judge
Direct LLM judge functionality for evaluating individual responses.

```bash
# Run example evaluation
python mitre_llm_judge.py

# Output:
# - Detailed scores for both responses
# - Reasoning for each score
# - Strengths and weaknesses
# - Comparative winner analysis
```

**Key Classes:**
- `LLMJudge`: Core judging functionality
- `BatchLLMEvaluator`: Evaluate multiple queries
- Utility functions for pretty-printing results

---

### 2. **mitre_rag_vs_graph_comparison.py** - Response Generation
Generates responses from both approaches (without judging).

```python
from mitre_rag_vs_graph_comparison import ComparisonEvaluator

evaluator = ComparisonEvaluator()
rag_metrics, graph_metrics = evaluator.evaluate_query(query)
```

---

### 3. **mitre_integrated_evaluation.py** ⭐ **START HERE**
Complete pipeline: Generate responses + Judge them + Generate report

```bash
# Run full integrated evaluation
python mitre_integrated_evaluation.py

# This will:
# 1. Query RAG approach
# 2. Query Graph+LLM approach
# 3. Have LLM judge both
# 4. Generate comprehensive report
# 5. Save results to JSON
```

---

## Quick Start: Using the Integrated Evaluator

### Basic Usage (Minimal Code)

```python
from mitre_integrated_evaluation import IntegratedLLMEvaluator

# Initialize
evaluator = IntegratedLLMEvaluator()

# Evaluate single query
result = evaluator.evaluate_single_query(
    "What techniques do threat actors use for credential theft?"
)

# Evaluate multiple queries
queries = [
    "How to detect lateral movement?",
    "What are persistence mechanisms?",
    "Which techniques does APT28 use?"
]
evaluator.evaluate_batch(queries)

# Generate report
print(evaluator.generate_report(include_full_results=True))

# Save results
evaluator.save_results()
```

### Output Structure

```json
{
  "timestamp": "2025-12-23 10:30:15",
  "summary": {
    "total_queries": 3,
    "rag_average": 7.2,
    "graph_average": 8.5,
    "improvement": 1.3,
    "graph_wins": 2,
    "rag_wins": 1,
    "ties": 0
  },
  "results": [
    {
      "query": "...",
      "rag": {
        "relevance": 7.5,
        "completeness": 6.5,
        "accuracy": 8.0,
        "specificity": 6.0,
        "clarity": 8.0,
        "overall": 7.2,
        "confidence": 0.9,
        "reasoning": "...",
        "strengths": [...],
        "weaknesses": [...]
      },
      "graph": {
        "relevance": 8.5,
        "completeness": 8.5,
        "accuracy": 8.0,
        "specificity": 8.5,
        "clarity": 8.0,
        "overall": 8.5,
        "confidence": 0.95,
        "reasoning": "...",
        "strengths": [...],
        "weaknesses": [...]
      },
      "comparison": {
        "winner": "Graph+LLM",
        "primary_reason": "...",
        "key_difference": "...",
        "recommendation": "...",
        "margin": 1.3
      },
      "performance": {
        "rag_latency_ms": 145.0,
        "graph_latency_ms": 298.0,
        "rag_entities": 5,
        "graph_entities": 23
      }
    }
  ]
}
```

---

## Example Evaluation Flow

### Step 1: Initialize
```python
from mitre_integrated_evaluation import IntegratedLLMEvaluator

evaluator = IntegratedLLMEvaluator()
# Output: Initializes database, embedding model, LLM judge
```

### Step 2: Query First Approach
```
RAG Approach (Semantic Search Only):
- Retrieves 5 most similar entities to query
- Generates context from those entities
- LLM generates response from focused context
```

### Step 3: Query Second Approach
```
Graph+LLM Approach (Semantic + Traversal):
- Retrieves 5 most similar entities (seeds)
- Traverses graph to find 2-hop connected entities
- Generates richer context from 20-50 entities total
- LLM generates response from enriched context
```

### Step 4: LLM Judge Evaluates Both
```
Judge Process:
1. Reads RAG response
2. Scores on 5 dimensions
3. Reads Graph response
4. Scores on same 5 dimensions
5. Compares and determines winner
6. Explains reasoning
```

### Step 5: Generate Report
```
Report includes:
- Individual scores for each response
- Comparison analysis
- Winner determination with reasoning
- Key strengths/weaknesses
- Final recommendation
```

---

## Interpreting Judge Scores

### What Different Scores Mean

| RAG Score | Graph Score | Interpretation |
|-----------|------------|-----------------|
| 7.5 | 8.5 | Graph+LLM clearly wins (+1.0 point) |
| 8.0 | 8.2 | Graph slightly better (marginal difference) |
| 7.8 | 7.8 | Tie - approaches equal for this query |
| 8.0 | 7.5 | RAG actually wins (Graph had too much noise) |

### Judge Confidence Score

The judge also reports confidence (0-1):
- **0.95-1.0**: Confident in assessment (clear winner)
- **0.8-0.95**: Reasonably confident (likely winner)
- **0.6-0.8**: Uncertain (close call)
- <0.6: Low confidence (very close or conflicting criteria)

**Use this to weight your final decision:**
```
If confidence >= 0.9:  Trust the winner strongly
If confidence 0.75-0.9: Trust the winner moderately
If confidence < 0.75:  Consider both approaches equal
```

---

## Common Patterns in Judge Scores

### Pattern 1: Graph Dominates (Graph Wins Decisively)
```
Query: "What techniques do threat actors use for credential theft?"

RAG:   Relevance 7, Completeness 6, Accuracy 8, Specificity 6, Clarity 8 → 7.0
Graph: Relevance 9, Completeness 9, Accuracy 8, Specificity 8, Clarity 8 → 8.4

Reason: More context → better completeness and specificity
Recommendation: Use Graph for complex analytical queries
```

### Pattern 2: RAG is Efficient (RAG Competitive)
```
Query: "What is T1055 (Process Injection)?"

RAG:   Relevance 9, Completeness 8, Accuracy 9, Specificity 9, Clarity 9 → 8.8
Graph: Relevance 9, Completeness 8, Accuracy 9, Specificity 9, Clarity 8 → 8.6

Reason: Simple query doesn't need graph expansion
Recommendation: Use RAG for direct definition/lookup queries
```

### Pattern 3: Graph Has Extra Noise (RAG Wins)
```
Query: "What is the first step in an attack?"

RAG:   Relevance 8, Completeness 7, Accuracy 9, Specificity 7, Clarity 8 → 7.8
Graph: Relevance 7, Completeness 6, Accuracy 8, Specificity 8, Clarity 7 → 7.2

Reason: Too much context confuses the answer
Recommendation: Use RAG for focused queries
```

---

## When to Use Each Approach

### Use Graph+LLM When:
✅ User asks complex questions (multi-clause, "how", "why")
✅ Need comprehensive threat intelligence
✅ Analyzing attack chains
✅ Incident investigation
✅ You have time flexibility (200-500ms acceptable)
✅ Quality matters more than speed

**Typical Judge Scores**: 8.0-9.0/10

### Use RAG When:
✅ Simple definition/lookup queries
✅ Need fast responses (<200ms)
✅ Bandwidth/cost is critical
✅ User knows exactly what they want
✅ Real-time applications
✅ Quick reference

**Typical Judge Scores**: 7.5-8.5/10

---

## Troubleshooting Judge Issues

### Problem: Judge Gives Very Low Scores (<5.0)
**Cause**: LLM is being overly critical
**Solution**:
```python
# Adjust judge temperature for more lenient scoring
judge = LLMJudge()
judge._call_llm(prompt, temperature=0.1)  # More conservative
```

### Problem: Judge Gives Same Score to Both
**Cause**: Responses are very similar or judge can't differentiate
**Solution**:
- Try more diverse test queries
- Run confidence score - it should be low if uncertain
- Manually review responses for differences judge might miss

### Problem: Judge Results Don't Match Your Assessment
**Cause**: Judge has different priorities than you
**Solution**:
```python
# Modify the judge prompt to emphasize your priorities
# In mitre_llm_judge.py, increase weight for your criteria
```

---

## Advanced: Custom Judge Configuration

### Focus on Specific Dimensions

```python
# If you only care about accuracy and relevance
class CustomJudge(LLMJudge):
    def evaluate_response(self, query, response, context_size=0, approach_name=""):
        # ... same evaluation ...
        # But modify scoring to weight accuracy/relevance higher
        
        overall = (accuracy * 0.5 + relevance * 0.5 +
                  completeness * 0.2 + specificity * 0.2 + clarity * 0.1)
        return overall
```

### Domain-Specific Rubrics

Modify the judge prompt to be more specific to your domain:

```python
# In evaluation_prompt, add domain-specific criteria
# Example for incident response:
"""
Additional for IR context:
- TTP Identification: How well does it identify technique IDs?
- Impact Assessment: Does it explain the impact?
- Mitigation Guidance: Are mitigations specific and actionable?
"""
```

---

## Batch Evaluation Best Practices

### Optimal Batch Size
- **Small batch (1-2 queries)**: Quick test, ~30 seconds
- **Medium batch (5-10 queries)**: Balanced, ~2-5 minutes
- **Large batch (20+ queries)**: Comprehensive, ~10+ minutes

### Query Diversity
For reliable results, include diverse queries:
```python
queries = [
    # Technique-focused
    "What is T1055?",
    # Threat actor-focused
    "What techniques does APT28 use?",
    # Defensive-focused
    "How to detect lateral movement?",
    # Complex multi-step
    "Analyze a ransomware attack chain",
    # Broad topics
    "What is privilege escalation?",
]
```

### Statistical Validity
For statistically valid results:
- **Minimum**: 5 queries per approach comparison
- **Recommended**: 10-20 queries
- **Comprehensive**: 50+ queries

The more queries you evaluate, the more confident you can be in the results.

---

## Expected Results Summary

### Typical Batch Evaluation (10 queries)

```
RAG Average:       7.4/10 ± 0.8
Graph+LLM Average: 8.5/10 ± 0.6

Quality Improvement: +1.1 points (+14.9%)

Win Record:
  Graph+LLM: 7 wins (70%)
  RAG:       2 wins (20%)
  Ties:      1 (10%)

Average Latency:
  RAG:       145ms
  Graph+LLM: 285ms

Recommendation: Use Graph+LLM for 70% of use cases
```

---

## Next Steps

1. **Run Basic Test**: `python mitre_integrated_evaluation.py`
2. **Review Results**: Check llm_judge_evaluation.json
3. **Evaluate More Queries**: Modify test queries list
4. **Analyze Patterns**: Look for when each approach wins
5. **Make Decision**: Which approach for your use case?

---

## See Also

- `RAG_VS_GRAPH_QUICK_START.md` - Quick reference guide
- `RAG_VS_GRAPH_EVALUATION_GUIDE.md` - Detailed metrics
- `mitre_automated_metrics.py` - Non-LLM evaluation metrics

