# RAG vs Graph+LLM Comparison Framework
## For MITRE ATT&CK Cybersecurity Queries

## Overview

This framework evaluates two approaches to generating responses for cybersecurity queries:

1. **Pure RAG (Retrieval Augmented Generation)**: Semantic search only → LLM
2. **Graph+LLM**: Semantic search + graph traversal → enriched context → LLM

## Test Queries for Evaluation

### Query Set 1: Technique Discovery
```
Query: "What techniques do threat actors use for credential theft?"

Expected RAG Results:
- T1110 (Brute Force)
- T1187 (Forced Authentication)
- T1056 (Input Capture)
- T1040 (Network Sniffing)
- T1056.004 (Credential API Hooking)

Expected Graph Expansion:
- Related threat actors (G0xxx groups that use these techniques)
- Related malware families that employ these tactics
- Detection strategies from MITRE
- Real-world usage from CISA advisories
```

### Query Set 2: Threat Actor Analysis
```
Query: "What are the common attack patterns used by APT28?"

Expected RAG Results:
- T1087 (Account Discovery)
- T1010 (Application Window Discovery)
- T1217 (Browser Bookmark Discovery)
- T1580 (Cloud Infrastructure Discovery)

Expected Graph Expansion:
- All techniques linked to APT28
- Related threat actors with similar TTPs
- Malware used by APT28
- Detection methods by MITRE
- CISA advisories mentioning APT28
```

### Query Set 3: Defensive Measures
```
Query: "How can we detect persistence mechanisms in Windows systems?"

Expected RAG Results:
- T1547 (Boot or Logon Autostart Execution)
- T1547.001 (Registry Run Keys / Startup Folder)
- T1547.014 (Active Setup)
- T1554 (Compromise Client Software Binary)

Expected Graph Expansion:
- Related persistence techniques
- Subtechniques for each parent technique
- Mitigation strategies from MITRE
- Detection rules from MITRE
- CISA advisories mentioning persistence
```

### Query Set 4: Real-World Incident Response
```
Query: "What should we look for if we suspect Emotet malware activity?"

Expected RAG Results:
- Techniques used by Emotet
- Related malware families
- Detection indicators

Expected Graph Expansion:
- Complete attack chain
- Threat actor groups using Emotet
- Victim industries and sectors
- Mitigation measures
- CISA advisories with Emotet mentions
```

## Evaluation Metrics

### 1. **Relevance Score** (0-10)
**Definition**: How directly does the response address the user's query?

**Scoring Guidelines**:
- **9-10**: Response directly answers query with highly relevant information
  - Example: Query about credential theft → immediately lists credential-related techniques
  
- **7-8**: Response is relevant with minor tangents
  - Example: Lists mostly credential techniques + some tangentially related ones
  
- **5-6**: Response somewhat relevant but mixed with irrelevant information
  - Example: Includes some credential techniques but also unrelated techniques
  
- **3-4**: Response has some relevant elements but mostly off-topic
  - Example: Mentions a few relevant techniques among many unrelated ones
  
- **0-2**: Response is not relevant to the query

**Assessment Method**:
- Read response carefully
- Count percentage of response content directly addressing the query
- Subtract points for tangential or incorrect information

---

### 2. **Completeness Score** (0-10)
**Definition**: Does the response cover the full scope of the topic?

**Scoring Guidelines**:
- **9-10**: Comprehensive coverage of all major aspects
  - Includes primary techniques, subtechniques, threat actors, detection, mitigation
  - For credential theft: covers brute force, credential access, input capture
  
- **7-8**: Covers most important aspects
  - Missing one category (e.g., no mitigation advice) but otherwise complete
  
- **5-6**: Covers multiple aspects but incomplete
  - Includes techniques and threat actors but no detection methods
  
- **3-4**: Covers basic information only
  - Lists techniques but no context or related information
  
- **0-2**: Minimal or surface-level information

**Assessment Method**:
- Evaluate breadth of coverage across: techniques, threat actors, detection, mitigation
- Compare to what a complete answer should include
- Graph approach should score higher due to expanded context

---

### 3. **Accuracy Score** (0-10)
**Definition**: Is the information factually correct according to MITRE ATT&CK?

**Scoring Guidelines**:
- **9-10**: All information accurate per MITRE ATT&CK framework
  - Correct technique IDs, names, descriptions
  - Accurate threat actor attributions
  - Correct relationships
  
- **7-8**: Mostly accurate with minor imprecisions
  - Slight misattributions or incomplete descriptions but not wrong
  
- **5-6**: Some inaccuracies or hallucinations present
  - A few incorrect statements or misnamed techniques
  
- **3-4**: Significant inaccuracies
  - Multiple wrong attributions or technique misidentifications
  
- **0-2**: Highly inaccurate or fabricated information

**Assessment Method**:
- Verify each technique ID against MITRE database
- Check threat actor attributions
- Look for hallucinations (made-up techniques or relationships)
- Cross-reference with actual MITRE ATT&CK data

---

## Performance Metrics (Automatically Collected)

### Latency (milliseconds)
- **RAG Expected**: 100-300 ms (semantic search + LLM generation)
- **Graph Expected**: 200-500 ms (semantic search + traversal + LLM generation)
- **Trade-off**: Graph is slower but provides richer context

### Context Size (number of entities)
- **RAG Expected**: 5-10 entities
- **Graph Expected**: 15-50 entities (with 1-2 hop expansion)
- **Ratio**: Graph provides 2-10x more context

### Token Usage
- **RAG Expected**: 100-300 tokens
- **Graph Expected**: 200-400 tokens
- Both approaches should be economical for LLM costs

### Entities Retrieved
- Tracked separately for comparison
- Graph should retrieve 3-5x more related entities

---

## Quick Scoring Template

For each query tested:

```markdown
## Query: [Your Query Here]

### RAG Approach
- Relevance: __ / 10
- Completeness: __ / 10  
- Accuracy: __ / 10
- **Average: __ / 10**

### Graph+LLM Approach
- Relevance: __ / 10
- Completeness: __ / 10
- Accuracy: __ / 10
- **Average: __ / 10**

### Performance
- RAG Latency: __ ms
- Graph Latency: __ ms
- RAG Entities: __
- Graph Entities: __

### Winner & Reasoning
[Explain which approach performed better and why]
```

---

## Expected Outcomes

### Hypothesis
Graph+LLM should outperform pure RAG because:

1. **Better Context**: Related techniques, threat actors, and mitigations provide fuller picture
2. **Higher Completeness**: Graph traversal discovers connections RAG misses
3. **Better Accuracy**: Graph relationships validate information consistency
4. **Richer Response**: LLM has more material to generate comprehensive answers

### Potential Trade-offs
- **Latency**: Graph traversal adds ~100-200ms overhead
- **Complexity**: Graph approach requires database traversal logic
- **Irrelevant Context**: If hop expansion is too deep, may include noise

### Success Criteria
- Graph+LLM achieves **+1.5 to +2.5 points** higher quality score
- Latency difference < 50% (acceptable trade-off)
- Completeness improvement > 20%
- Accuracy comparable between approaches

---

## Running the Comparison

### Basic Usage
```bash
# Run with default queries
python mitre_rag_vs_graph_comparison.py

# Will output:
# - Side-by-side query execution
# - Responses from both approaches
# - Automatic performance metrics
# - Manual scoring prompts
```

### Detailed Scoring Workflow

1. **Run first query** and review both responses
2. **Score each dimension** (relevance, completeness, accuracy) on 0-10 scale
3. **Consider context**: Is more information better? Is it relevant?
4. **Note trade-offs**: Is the extra latency worth the quality gain?
5. **Repeat for multiple queries** to get statistically valid results

### Interpreting Results

**If Graph+LLM Wins**:
- More comprehensive queries are better for threat intelligence
- Recommendation: Use Graph approach for enterprise threat analysis

**If RAG Wins**:
- Simpler queries work better with focused context
- Recommendation: Use RAG for quick lookups, Graph for deep analysis

**If Tied**:
- Choose based on performance requirements
- RAG for speed-critical applications
- Graph for quality-critical applications

---

## Example Scoring Session

```python
# Run with manual scores
evaluator.evaluate_query(
    "What techniques are used for lateral movement?",
    manual_scores={
        'rag': {
            'relevance': 8.0,    # Directly answers about lateral movement
            'completeness': 6.5, # Missing some advanced persistence techniques
            'accuracy': 9.0      # All techniques correct
        },
        'graph': {
            'relevance': 8.5,    # Excellent answer with context
            'completeness': 8.5, # Includes detection methods + related techniques
            'accuracy': 9.0      # Accurate throughout
        }
    }
)

# Output shows:
# - Graph improvement: +0.67 points (6.7%)
# - Both have good accuracy
# - Graph provides better completeness
```

---

## Metrics Summary Table

| Metric | RAG | Graph+LLM | Winner | Importance |
|--------|-----|-----------|--------|------------|
| Relevance | 7-8 | 8-9 | Graph | Very High |
| Completeness | 6-7 | 8-9 | Graph | Very High |
| Accuracy | 8-9 | 8-9 | Tie | Critical |
| Latency | 100-300ms | 200-500ms | RAG | Medium |
| Context Size | 5-10 | 15-50 | Graph | Medium |
| Token Usage | 100-300 | 200-400 | RAG | Low |

---

## Next Steps

1. **Run 5-10 queries** through both approaches
2. **Score each systematically** using the template
3. **Analyze patterns**: Which query types benefit most from graph traversal?
4. **Generate report**: Use `evaluator.generate_report()` to see aggregated results
5. **Determine threshold**: What quality improvement is worth the latency cost?

