# RAG vs Graph+LLM Comparison: Quick Start Guide

## What's Included

Three Python modules for comprehensive comparison:

### 1. **mitre_rag_vs_graph_comparison.py** ⭐ START HERE
Main evaluation framework with two approaches:
- **PureRAGApproach**: Semantic search only → LLM
- **GraphSearchApproach**: Semantic search + graph traversal → LLM

### 2. **mitre_automated_metrics.py** 
Objective metrics (no manual scoring needed):
- BLEU, ROUGE-L, Information Density
- Coverage, Hallucination Detection, Specificity
- Composite quality scores

### 3. **RAG_VS_GRAPH_EVALUATION_GUIDE.md**
Comprehensive guide including:
- Detailed evaluation metrics
- Scoring rubrics (0-10)
- Test queries
- Expected outcomes

---

## Quick Start (5 minutes)

### Step 1: Verify Dependencies
```bash
pip install sentence-transformers arango requests
```

### Step 2: Verify Database Connection
```bash
# Check if ArangoDB is running
curl -u root:openSesame http://localhost:8529

# Check if Ollama is running  
curl http://localhost:11434/api/tags
```

### Step 3: Run Basic Comparison
```bash
python mitre_rag_vs_graph_comparison.py
```

This will:
1. Run one test query through both approaches
2. Show responses side-by-side
3. Collect automatic performance metrics
4. Save results to `comparison_results.json`

---

## Understanding the Output

### Response Comparison Example

```
===========================================================================
Query: What techniques do threat actors use for credential theft?
===========================================================================

[1] Pure RAG Approach (Semantic Search Only)
  - Retrieved: 5 entities
  - Latency: 145.32 ms
  - Context size: 342 chars
  - Response tokens: 187

  Response preview: For credential theft attacks, threat actors commonly use...

[2] Graph+LLM Approach (Semantic + Traversal)
  - Seed entities: 5
  - Connected entities: 18
  - Total context: 23 entities
  - Latency: 298.41 ms
  - Context size: 1246 chars
  - Response tokens: 205

  Response preview: Credential theft techniques in MITRE ATT&CK include...

[3] Manual Evaluation Scores (0-10)

  RAG Approach Scores:
    - Relevance: 7.5/10
    - Completeness: 6.5/10
    - Accuracy: 8.0/10
    - Overall: 7.3/10

  Graph+LLM Approach Scores:
    - Relevance: 8.5/10
    - Completeness: 8.5/10
    - Accuracy: 8.5/10
    - Overall: 8.5/10

[4] Comparison Results
  - Quality improvement (Graph vs RAG): +1.2 points (+16.4%)
  - Latency difference: +153.09 ms (+105.3%)
  - Context expansion: +360.5%
```

### Key Takeaways
- **Quality**: Graph+LLM won by 1.2 points (16.4% improvement)
- **Speed Trade-off**: Graph is ~153ms slower but provides 4x more context
- **Context**: Graph provides 23 entities vs 5 from RAG
- **Recommendation**: Use Graph for comprehensive analysis, RAG for quick queries

---

## Advanced Usage

### Run Multiple Queries with Scoring

```python
from mitre_rag_vs_graph_comparison import ComparisonEvaluator

evaluator = ComparisonEvaluator()

queries = [
    "What techniques are used for lateral movement?",
    "How can we detect persistence mechanisms?",
    "What tools does APT28 typically use?"
]

for query in queries:
    # Score based on your assessment
    scores = {
        'rag': {
            'relevance': 7.5,      # 0-10 scale
            'completeness': 6.5,
            'accuracy': 8.0
        },
        'graph': {
            'relevance': 8.5,
            'completeness': 8.5,
            'accuracy': 8.5
        }
    }
    
    evaluator.evaluate_query(query, scores)

# Generate final report
print(evaluator.generate_report())
```

### Use Automated Metrics (No Manual Scoring)

```python
from mitre_automated_metrics import ComparisonMetricsCalculator

calculator = ComparisonMetricsCalculator(
    known_techniques=['T1110', 'T1187', 'T1056', 'T1040']
)

metrics = calculator.calculate_all(
    query="Credential theft techniques",
    rag_response="RAG response text...",
    graph_response="Graph+LLM response text...",
    expected_topics=['T1110', 'threat actors', 'mitigation'],
    rag_context_size=50,
    graph_context_size=150
)

# Print results
ComparisonMetricsCalculator.print_metrics(metrics)
```

---

## Metric Interpretation Guide

### Relevance (0-10)
- **9-10**: Directly answers the query with no tangents
- **7-8**: Relevant with minor tangential information
- **5-6**: Somewhat relevant, mixed with unrelated info
- **3-4**: Minimally relevant
- **0-2**: Not relevant

### Completeness (0-10)
- **9-10**: Covers all major aspects (techniques + actors + detection + mitigation)
- **7-8**: Covers most aspects
- **5-6**: Covers some aspects
- **3-4**: Minimal coverage
- **0-2**: Surface-level only

### Accuracy (0-10)
- **9-10**: All facts correct per MITRE ATT&CK
- **7-8**: Mostly correct with minor imprecisions
- **5-6**: Some inaccuracies present
- **3-4**: Multiple errors
- **0-2**: Highly inaccurate

### Automatic Metrics (0-1 scale)

**BLEU Score**: Measures n-gram overlap
- 0.8-1.0: Excellent alignment
- 0.6-0.8: Good alignment
- 0.4-0.6: Partial alignment
- <0.4: Poor alignment

**ROUGE-L Score**: Measures longest common subsequence
- 0.8-1.0: Very similar content
- 0.6-0.8: Similar content
- 0.4-0.6: Some similarity
- <0.4: Different content

**Information Density**: Unique info per context unit
- 0.8-1.0: Very efficient
- 0.6-0.8: Efficient
- 0.4-0.6: Moderate efficiency
- <0.4: Low efficiency

**Coverage**: Percentage of expected topics mentioned
- 0.8-1.0: Excellent coverage
- 0.6-0.8: Good coverage
- 0.4-0.6: Partial coverage
- <0.4: Poor coverage

**Hallucination Score**: Valid entities / total mentioned
- 0.9-1.0: Trustworthy (no hallucinations)
- 0.7-0.9: Mostly accurate
- 0.5-0.7: Some concerns
- <0.5: Unreliable

---

## Expected Results

### Hypothesis
Graph+LLM should win on quality due to richer context.

### Typical Outcomes

| Metric | RAG | Graph | Winner |
|--------|-----|-------|--------|
| Relevance | 7-8 | 8-9 | Graph |
| Completeness | 6-7 | 8-9 | Graph |
| Accuracy | 8-9 | 8-9 | Tie |
| Latency | 100-300ms | 200-500ms | RAG |
| Context Size | 5-10 ents | 15-50 ents | Graph |
| **Overall Quality** | **7.2/10** | **8.5/10** | **Graph** |

### Success Criteria
- ✅ Graph wins on quality by 1-2 points
- ✅ Latency overhead < 50% difference
- ✅ Completeness improves > 20%
- ✅ Accuracy stays above 8.0/10

---

## Troubleshooting

### Database Connection Error
```
Error: Connection refused at localhost:8529
```

**Solution**: Start ArangoDB
```bash
docker run -d -p 8529:8529 -e ARANGO_ROOT_PASSWORD=openSesame arangodb
```

### Ollama Connection Error
```
Error generating response: Connection refused at localhost:11434
```

**Solution**: Start Ollama
```bash
ollama serve &
ollama pull llama3.1:8b
```

### Out of Memory
If graph traversal times out or runs out of memory:
```python
# Reduce traversal depth
graph_approach.search(query, top_k=3, max_depth=1)

# Or reduce context size
top_k = 3  # Fewer seed entities
```

### Slow Performance
If latency > 500ms:
```python
# Check database indices
db.collection('entity_embeddings').properties()

# Verify Ollama model loaded
curl http://localhost:11434/api/tags
```

---

## Recommended Workflow

### Phase 1: Exploration (10 minutes)
1. Run basic comparison on 1-2 queries
2. Review responses from both approaches
3. Understand trade-offs

### Phase 2: Systematic Evaluation (30-60 minutes)
1. Select 5-10 representative queries
2. Run through both approaches
3. Score manually using the rubric
4. Calculate automated metrics
5. Generate comparison report

### Phase 3: Analysis (15 minutes)
1. Review overall report
2. Identify patterns
3. Determine if Graph+LLM worth the latency
4. Make recommendation

### Phase 4: Optimization (Optional)
1. Fine-tune traversal depth
2. Optimize context selection
3. Benchmark with production queries

---

## Query Selection Tips

**Good Test Queries:**
- ✅ "What techniques are used for lateral movement?"
- ✅ "How does APT28 operate?"
- ✅ "What are common persistence mechanisms?"
- ✅ "How can we detect credential theft?"
- ✅ "What tools does Lazarus use?"

**Avoid:**
- ❌ Yes/no questions (not enough context for evaluation)
- ❌ Too specific (e.g., "T1055.004 definition" - not enough to compare)
- ❌ Out of domain (e.g., "How to bake a cake")

---

## Output Files

After running evaluation:

**comparison_results.json**: 
```json
[
  {
    "query": "What techniques...",
    "approach": "rag",
    "response": "...",
    "latency_ms": 145.32,
    "context_size": 5,
    "relevance_score": 7.5,
    "completeness_score": 6.5,
    "accuracy_score": 8.0,
    "total_score": 7.33,
    "efficiency": 0.0345
  },
  ...
]
```

**Automated Report** (from `generate_report()`):
Shows aggregated statistics across all queries

---

## Integration with Your System

### Use Results to Decide Approach

**Choose RAG if:**
- Response time is critical (<200ms required)
- Users ask simple lookup questions
- You need minimum latency
- Cost per query is critical

**Choose Graph+LLM if:**
- Quality is more important than speed
- Users ask complex analytical questions
- You want comprehensive context
- Users are willing to wait 200-500ms

**Hybrid Approach:**
```python
# Use Graph for complex queries, RAG for simple ones
if len(query.split()) > 10 and "how" in query:
    use_graph_approach()  # Complex analysis
else:
    use_rag_approach()    # Quick lookup
```

---

## Example Results Summary

```
Query Evaluation Summary (5 queries tested)
==============================================

Overall Winner: Graph+LLM Approach ⭐
Average Quality Score:  7.1/10 (RAG) vs 8.3/10 (Graph)
Improvement: +1.2 points (+16.9%)

Best For Complex Analysis: Graph+LLM
- Better completeness (+1.8 points)
- Better coverage (+22%)
- Slightly higher latency (-153ms acceptable)

Best For Speed: RAG
- 50% faster (153ms average)
- Good accuracy maintained
- Suitable for real-time queries

Recommendation: Use Graph+LLM for threat intelligence
analysis and RAG for quick reference lookups.
```

---

## Next Steps

1. **Run basic test**: `python mitre_rag_vs_graph_comparison.py`
2. **Review output**: Check response quality
3. **Score systematically**: Use 5-10 test queries
4. **Generate report**: `evaluator.generate_report()`
5. **Make decision**: Which approach for your use case?

---

**Need help?** Check `RAG_VS_GRAPH_EVALUATION_GUIDE.md` for detailed metrics definitions.
