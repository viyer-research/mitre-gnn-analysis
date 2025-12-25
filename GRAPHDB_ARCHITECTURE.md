# MITRE2KG GRAPH DATABASE ARCHITECTURE

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────┐        ┌──────────────────────────────┐  │
│  │  MITRE ATT&CK        │        │  CISA Cybersecurity         │  │
│  │  Enterprise JSON     │        │  Advisories CSV             │  │
│  │                      │        │                              │  │
│  │ • Attack Patterns    │        │ • 77 Advisories            │  │
│  │ • Tactics            │        │ • 432 Unique T#            │  │
│  │ • Threat Groups      │        │ • Real-world attacks       │  │
│  │ • Malware/Tools      │        │ • Sector-specific intel    │  │
│  │ • Relationships      │        │                              │  │
│  └──────────┬───────────┘        └────────────┬─────────────────┘  │
│             │                                  │                    │
└─────────────┼──────────────────────────────────┼────────────────────┘
              │                                  │
              │ PARSE & EXTRACT                 │ PARSE & EXTRACT
              │ TRIPLETS                        │ TRIPLETS
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA TRANSFORMATION                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────┐  ┌────────────────────────────┐   │
│  │  Entity Extraction          │  │  Relationship Extraction   │   │
│  │                             │  │                            │   │
│  │ T1055 - Process Injection  │  │ T1055 ─subtechnique-of─    │   │
│  │ T1055.004 - Extra Memory   │  │ T1055.004                  │   │
│  │ TA0002 - Execution         │  │                            │   │
│  │ G0001 - APT1               │  │ T1055 ─belongs-to-tactic─  │   │
│  │ aa24-060a - Phobos Advisory│  │ TA0002                     │   │
│  │ ...                         │  │                            │   │
│  │                             │  │ G0001 ─uses─ T1055        │   │
│  │ 24,556 Total Entities       │  │                            │   │
│  └────────────┬────────────────┘  │ 24,342 Relationships       │   │
│               │                    └─────────┬──────────────────┘   │
└───────────────┼────────────────────────────────┼───────────────────┘
                │                                 │
                └──────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ARANGODB STORAGE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────┐ │
│  │  entities          │  │  relationships     │  │  embeddings  │ │
│  │                    │  │                    │  │              │ │
│  │ Type: Document     │  │ Type: Edge         │  │ Vectors:     │ │
│  │ Count: 24,556      │  │ Count: 24,342      │  │ 47,293       │ │
│  │                    │  │                    │  │ Dims: 384    │ │
│  │ • Attack-pattern   │  │ • uses (17,270)    │  │              │ │
│  │ • Malware (695)    │  │ • uses_in_cisa    │  │ Models:      │ │
│  │ • Intrusion-set    │  │ • mitigates        │  │ all-MiniLM   │ │
│  │ • Campaign         │  │ • detects          │  │ L6-v2        │ │
│  │ • CISA_advisory    │  │ • belongs_to_tactic│  │              │ │
│  │ • Tool             │  │ • subtechnique-of  │  │ • entity_    │ │
│  │ • Analytic         │  │ • attributed-to    │  │   embeddings │ │
│  │                    │  │ • revoked-by       │  │   (24,556)   │ │
│  │                    │  │ • referenced_in    │  │              │ │
│  │                    │  │   cisa_advisory    │  │ • relationship│ │
│  │                    │  │                    │  │   embeddings│ │
│  │                    │  │                    │  │   (22,737)   │ │
│  └────────────────────┘  └────────────────────┘  └──────────────┘ │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           │ AQL Queries & Traversal
                           │
┌──────────────────────────┴──────────────────────────────────────────┐
│                     APPLICATION LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐   │
│  │  RAG Query Engine    │  │  Semantic Search                 │   │
│  │  (arangodb_rag_      │  │  (Vector Similarity)             │   │
│  │   query_engine.py)   │  │                                  │   │
│  │                      │  │ • Query embedding → encode      │   │
│  │ • Graph traversal    │  │ • Find nearest neighbors        │   │
│  │ • Context retrieval  │  │ • Cosine similarity             │   │
│  │ • Multi-hop paths    │  │ • Return top-k results          │   │
│  └──────────┬───────────┘  └────────────┬─────────────────────┘   │
│             │                            │                         │
│             └────────────┬────────────────┘                         │
│                          ▼                                          │
│              ┌──────────────────────┐                               │
│              │  Ollama LLM          │                               │
│              │  (llama3.1:8b)       │                               │
│              │                      │                               │
│              │ • Query understanding│                               │
│              │ • Context integration│                               │
│              │ • Response generation│                               │
│              └──────────┬───────────┘                               │
│                         │                                          │
└─────────────────────────┼──────────────────────────────────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │  Final Response      │
              │  (Threat Intel)      │
              └──────────────────────┘
```

## Data Flow: From CSV to Graph

### CISA Advisory Integration Pipeline

```
CISA-crawl-rt-ttp-ct.csv
│
├─ Row 1: URL=aa24-060a, TTP={T1083, T1071.002, ...}
│
├─ Extract Advisory ID (aa24-060a)
│
├─ Create Entity:
│  └─ cisa_advisory(aa24-060a)
│
├─ Extract TTPs from field
│  └─ [T1083, T1071.002, T1055.004, T1657, ...]
│
├─ For each TTP:
│  ├─ Find attack-pattern entity (T1083)
│  └─ Create Relationship:
│     └─ attack-pattern(T1083) ──uses_in_cisa_advisory──> cisa_advisory(aa24-060a)
│
└─ Regenerate embeddings for updated techniques

Result: 1,605 new relationships across 368 techniques linking to 74 advisories
```

## Entity Relationship Diagram

```
                    ┌─────────────┐
                    │   TACTIC    │
                    │  (TA0002)   │
                    └──────▲──────┘
                           │ belongs-to-tactic
                           │
    ┌──────────────────────┴────────────────────┐
    │                                            │
    ▼                                            ▼
┌─────────────┐  subtechnique-of      ┌──────────────┐
│ TECHNIQUE   │ ◄────────────────────  │  SUB-TECHNIQUE
│  (T1055)    │ ─────────────────────► │  (T1055.004)
└──────┬──────┘                        └──────────────┘
       │ uses
       │ │ detects
       │ │ mitigates
       │ ▼
    ┌──────────────┐    attributed-to   ┌──────────────┐
    │   GROUP      │ ◄─────────────────  │  INTRUSION   │
    │   (G0001)    │                     │  SET         │
    └──────────────┘                     └──────────────┘
       │
       │ uses_in_cisa_advisory
       │ referenced_in_cisa_advisory
       │
       ▼
    ┌──────────────────┐
    │  CISA_ADVISORY   │
    │  (aa24-060a)     │
    │  "Phobos RW"     │
    └──────────────────┘
```

## Schema Statistics Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    DATABASE METRICS                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ENTITIES                          RELATIONSHIPS            │
│  ├─ Attack-patterns: 823           ├─ uses: 17,270         │
│  ├─ Malware: 695                   ├─ uses_in_cisa: 1,605  │
│  ├─ Intrusion-sets: 178            ├─ mitigates: 1,445     │
│  ├─ Campaigns: 52                  ├─ referenced_in: 1,619 │
│  ├─ Tools: 91                      ├─ belongs-to: 1,071    │
│  ├─ Analytic: 1,739                ├─ detects: 691         │
│  ├─ CISA advisories: 74            ├─ subtechnique: 476    │
│  ├─ Other: 18,264                  ├─ revoked-by: 140      │
│  │                                  ├─ attributed-to: 25    │
│  TOTAL: 24,556 entities            TOTAL: 24,342 edges     │
│                                                              │
│  EMBEDDINGS                                                 │
│  ├─ Entity vectors: 24,556 (384-dim)                       │
│  ├─ Relationship vectors: 22,737 (384-dim)                │
│  ├─ Total vectors: 47,293                                  │
│  ├─ Memory usage: ~69.3 MB                                │
│  └─ Model: all-MiniLM-L6-v2                               │
│                                                              │
│  CISA INTEGRATION METRICS                                  │
│  ├─ Advisories processed: 77                              │
│  ├─ Advisories in DB: 74                                  │
│  ├─ Unique T# techniques: 432                             │
│  ├─ Linked techniques: 368                                │
│  ├─ Total links created: 1,605                            │
│  └─ Success rate: 94.4%                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Query Examples

### Semantic Search
```python
# Find techniques similar to "stealing credentials"
query_embedding = embed("steal credentials").vector
results = db.semantic_search(query_embedding, top_k=10)
# Returns: T1528, T1110, T1187, T1111, ...
```

### Graph Traversal
```python
# Find all advisories that mention T1083 (File and Directory Discovery)
path = T1083 ──uses_in_cisa_advisory──> cisa_advisory
results = db.traverse(T1083, relationships=["uses_in_cisa_advisory"])
# Returns: [aa24-060a, aa24-038a, aa23-165a, ...]
```

### Multi-hop Queries
```python
# Find threat groups using techniques mentioned in ransomware advisories
path = cisa_advisory(ransomware) ◄──referenced_in── attack-pattern
       ◄──uses── group
# Returns threat actors involved in ransomware campaigns
```

## Integration Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    RAG + LLM PIPELINE                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query: "What techniques are used in phishing attacks?"   │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────────────┐                                       │
│  │  Encode Query       │  all-MiniLM-L6-v2                    │
│  │  (Embedding)        │  → 384-dim vector                    │
│  └────────────┬────────┘                                       │
│               │                                                │
│               ▼                                                │
│  ┌─────────────────────────────────────────┐                 │
│  │  Semantic Search in Embedding Space     │                 │
│  │  Find nearest neighbors to query        │                 │
│  │  ↓                                       │                 │
│  │  Results: T1566, T1566.002, T1598, ...  │                 │
│  └────────────┬────────────────────────────┘                 │
│               │                                                │
│               ▼                                                │
│  ┌─────────────────────────────────────────┐                 │
│  │  Graph Context Retrieval                │                 │
│  │  For each technique:                    │                 │
│  │  - Get CISA advisories                  │                 │
│  │  - Get threat groups                    │                 │
│  │  - Get mitigations                      │                 │
│  │  - Get detection strategies             │                 │
│  └────────────┬────────────────────────────┘                 │
│               │                                                │
│               ▼                                                │
│  ┌─────────────────────────────────────────┐                 │
│  │  LLM Prompt Construction                │                 │
│  │  Query + Context + Instructions         │                 │
│  │  ↓                                       │                 │
│  │  "Based on these techniques and        │                 │
│  │   real-world advisories, explain..."   │                 │
│  └────────────┬────────────────────────────┘                 │
│               │                                                │
│               ▼                                                │
│  ┌──────────────────────────────────────────┐                │
│  │  Ollama LLM (llama3.1:8b)                │                │
│  │  localhost:11434                         │                │
│  │  ↓                                        │                │
│  │  Generate coherent response              │                │
│  └────────────┬─────────────────────────────┘                │
│               │                                                │
│               ▼                                                │
│  ┌──────────────────────────────────────────┐                │
│  │  Final Answer with Citations             │                │
│  │  "Phishing attacks use T1566 (Phishing)  │                │
│  │   as mentioned in advisories aa24-060a,  │                │
│  │   aa23-165a, ... Mitigations include..."│                │
│  └──────────────────────────────────────────┘                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Key Features

✅ **Unified Knowledge Graph**
- MITRE ATT&CK framework + CISA real-world advisories
- 823 attack techniques with complete relationships

✅ **Semantic Search**
- 384-dimensional embeddings for all entities
- Find similar techniques by meaning, not keywords
- Cosine similarity in embedding space

✅ **Multi-hop Graph Traversal**
- Find related entities: techniques → groups → campaigns
- Build threat landscapes and attack chains
- Discover hidden connections

✅ **LLM Integration**
- Query understanding via Ollama
- Context injection from graph
- Natural language threat intelligence

✅ **Real-world Integration**
- 77 CISA cybersecurity advisories
- 1,605 technique-to-advisory links
- Current threat landscape

✅ **Scalable Architecture**
- ArangoDB for flexible querying
- Document + graph hybrid model
- Fast lookups and traversals
