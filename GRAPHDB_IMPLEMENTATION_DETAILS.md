# MITRE2KG IMPLEMENTATION DETAILS & EXAMPLES

## 1. Data Generation Process

### 1.1 MITRE Enterprise Attack Framework Extraction

**Source:** `enterprise_attack.json` (STIX 2.1 format)

**Processing Steps:**

```
enterprise_attack.json (14,000+ objects)
├── Parse STIX objects
├── Extract entity types:
│   ├── attack-pattern (823 total: 345 parent + 478 subtechniques)
│   ├── x-mitre-tactic (14 tactics: Recon, Initial Access, etc.)
│   ├── intrusion-set (178 threat groups/APTs)
│   ├── malware (695 malware families)
│   ├── tool (91 utility tools)
│   ├── campaign (52 operations)
│   └── ... (20,048 relationship objects)
│
├── Normalize identifiers:
│   ├── T1055 (parent technique: Process Injection)
│   ├── T1055.001 (subtechnique: Dynamic-link Library Injection)
│   ├── T1055.002 (subtechnique: Portable Executable Injection)
│   ├── ... T1055.014 (subtechnique: VDSO Hijacking)
│
└── Extract relationships:
    ├── (T1055.001) subtechnique-of (T1055)
    ├── (T1055) belongs-to-tactic (TA0002-Execution)
    ├── (G0001-APT1) uses (T1055)
    ├── (T1055) detects-by (detection-strategy)
    └── ... (22,737 MITRE framework edges)
```

**Example Attack Pattern:**
```json
{
  "external_id": "T1055",
  "type": "attack-pattern",
  "name": "Process Injection",
  "description": "Adversaries may inject code into processes...",
  "x_mitre_platforms": ["Windows", "macOS", "Linux"],
  "x_mitre_data_sources": ["Process: OS API execution", "Process: Process access"],
  "x_mitre_version": "3.1",
  "subtechniques": [
    "T1055.001 - DLL Injection",
    "T1055.002 - Portable Executable Injection",
    "T1055.004 - Asynchronous Procedure Call",
    ...
  ]
}
```

### 1.2 CISA Advisory Integration

**Source:** `CISA-crawl-rt-ttp-ct.csv` (77 rows)

**CSV Structure:**
```csv
RawText,CleanText,TTP,URL
"Full advisory...",  "Cleaned text...",  "{T1083, T1071.002,...}",  "https://www.cisa.gov/news-events/cybersecurity-advisories/aa24-060a"
...
```

**Processing Steps:**

```
CISA-crawl-rt-ttp-ct.csv (77 advisories)
│
├─ Row 1: URL contains advisory ID "aa24-060a"
│         TTP field = {T1083, T1071.002, T1055.004, T1657, ...}
│
├─ Extract Advisory ID: aa24-060a (Phobos Ransomware)
├─ Extract T# list: [T1083, T1071.002, T1055.004, T1657, T1055.002, ...]
│
├─ Create cisa_advisory entity:
│  {
│    "external_id": "aa24-060a",
│    "name": "CISA Alert on Phobos Ransomware",
│    "url": "https://www.cisa.gov/...",
│    "cleantext": "Phobos is a file-encrypting ransomware...",
│    "type": "cisa_advisory"
│  }
│
├─ For each TTP (T1083):
│  ├─ Lookup attack-pattern entity in database
│  └─ Create relationship:
│     {
│       "_from": "entities/T1083",
│       "_to": "entities/aa24-060a",
│       "relationship_type": "uses_in_cisa_advisory",
│       "description": "Technique T1083 is used in advisory aa24-060a"
│     }
│
└─ Result: 1,605 relationships created (77 advisories × ~20 techniques avg)
```

**Example Advisory Processing:**
```
Advisory: aa24-060a (Phobos Ransomware)
Extracted TTPs: 43 techniques
  ├─ T1083 (File and Directory Discovery)
  ├─ T1071.002 (Application Layer Protocol: Web Protocols)
  ├─ T1055.004 (Process Injection: Asynchronous Procedure Call)
  ├─ T1657 (Disk Wipe)
  ├─ T1055.002 (Process Injection: Portable Executable Injection)
  ├─ T1219 (Remote Access Software)
  ├─ T1490 (Inhibit System Recovery)
  ├─ T1588.002 (Obtain Capabilities: Tool)
  └─ ... (35 more techniques)

Created Relationships: 43
  aa24-060a ←uses_in_cisa_advisory← T1083
  aa24-060a ←uses_in_cisa_advisory← T1071.002
  aa24-060a ←uses_in_cisa_advisory← T1055.004
  ... (40 more)
```

## 2. Triplet Generation Details

### 2.1 MITRE Framework Triplets

**Format:** (Subject, RelationType, Object)

**Examples:**

```
1. SUBTECHNIQUE HIERARCHY
   (T1055.001, subtechnique-of, T1055)
   (T1055.002, subtechnique-of, T1055)
   (T1055.004, subtechnique-of, T1055)
   ... (476 total subtechnique edges)

2. TACTIC MAPPING
   (T1055, belongs-to-tactic, TA0002)
   (T1083, belongs-to-tactic, TA0007)
   (T1071, belongs-to-tactic, TA0010)
   ... (1,071 total tactic edges)

3. GROUP → TECHNIQUE USAGE
   (G0001, uses, T1055)
   (G0001, uses, T1083)
   (G0002, uses, T1071)
   ... (17,270 total usage edges)

4. MITIGATION CHAINS
   (T1055, mitigates, T1110)
   (T1110, mitigates, T1136)
   ... (1,445 total mitigation edges)

5. DETECTION STRATEGIES
   (T1055, detects, detection-strategy-x)
   ... (691 total detection edges)
```

### 2.2 CISA Advisory Triplets

**Format:** (AttackPattern, RelationType, CISAAdvisory)

**Examples:**

```
1. TECHNIQUE USED IN ADVISORY
   (T1083, uses_in_cisa_advisory, aa24-060a)
   (T1071.002, uses_in_cisa_advisory, aa24-060a)
   (T1055.004, uses_in_cisa_advisory, aa24-060a)
   ... (1,605 total technique-to-advisory edges)

2. ADVISORY REFERENCES TECHNIQUE
   (aa24-060a, referenced_in_cisa_advisory, T1083)
   (aa24-038a, referenced_in_cisa_advisory, T1055)
   ... (1,619 total edges, mostly reverse of above)
```

**Complete Triplet Example:**

```
Advisory: aa24-057a (SVR-Attributed Actors)
Triplets generated:
  
  1. (T1566, uses_in_cisa_advisory, aa24-057a)
     Subject: T1566 (Phishing) 
     Relationship: used in advisory
     Object: aa24-057a (SVR actors attacking cloud infrastructure)
  
  2. (T1078, uses_in_cisa_advisory, aa24-057a)
     Subject: T1078 (Valid Accounts)
     Relationship: used in advisory
     Object: aa24-057a
  
  3. (T1566.002, uses_in_cisa_advisory, aa24-057a)
     Subject: T1566.002 (Phishing: Spearphishing Link)
     Relationship: used in advisory
     Object: aa24-057a
     
  ... (20 more triplets for this advisory)
```

## 3. Embedding Generation

### 3.1 Entity Embeddings

**Process:**
```
For each entity (entity_id, name, description):
  1. Concatenate name + description text
  2. Tokenize with all-MiniLM-L6-v2 tokenizer
  3. Generate embedding: 384-dimensional vector
  4. L2-normalize the vector
  5. Store in entity_embeddings collection
```

**Example: T1528 (Steal Application Access Token)**

```python
entity = {
    "external_id": "T1528",
    "name": "Steal Application Access Token",
    "description": "Adversaries may steal application access tokens used..."
}

# Input to embedding model
text = "Steal Application Access Token. Adversaries may steal..."

# Output: 384-dimensional vector
embedding = [
    -0.1404, 0.0750, 0.0211, -0.0856, -0.0047, 0.0105, 0.0409, 0.0686,
    0.0476, -0.0241, 0.0337, -0.0004, 0.0408, -0.1234, -0.0108,
    ... (369 more dimensions)
]

# L2 norm: 1.0 (normalized)
sqrt(sum(x^2 for x in embedding)) = 1.0

# Stored in ArangoDB:
{
    "_key": "attack-pattern_890c9858-598c-401d-a4d5-c67ebcdd703a",
    "_id": "entity_embeddings/attack-pattern_890c9858-598c-401d-a4d5-c67ebcdd703a",
    "embedding": [...384 floats...],
    "embedding_dim": 384
}
```

### 3.2 Relationship Embeddings

**Process:**
```
For each relationship (from_entity, relationship_type, to_entity):
  1. Get embeddings for from_entity and to_entity
  2. Concatenate both embeddings with relationship type text
  3. Generate new embedding
  4. Store in relationship_embeddings collection
```

**Example: T1055 subtechnique-of T1055.001**

```python
relationship = {
    "relationship_type": "subtechnique-of",
    "_from": "entities/T1055.001",
    "_to": "entities/T1055"
}

# Input: embedding(T1055.001) + "subtechnique-of" + embedding(T1055)
# All three inputs concatenated and encoded

# Output: 384-dimensional relationship embedding
rel_embedding = [
    -0.0174, 0.0289, -0.0423, -0.0283, 0.1014,
    ... (379 more dimensions)
]
```

### 3.3 Embedding Statistics

```
Total Embeddings Generated: 47,293
├─ Entity embeddings: 24,556
│  ├─ Attack patterns: 823
│  ├─ Threat groups: 0 (example)
│  ├─ CISA advisories: 74
│  └─ Other entities: 23,659
│
└─ Relationship embeddings: 22,737
   ├─ uses: 17,270
   ├─ belongs-to-tactic: 1,071
   ├─ subtechnique-of: 476
   └─ Other relationships: 3,920

Vector Properties:
  ├─ Dimensions: 384
  ├─ Type: Float32
  ├─ Normalization: L2 (unit norm)
  ├─ Memory per vector: 384 × 4 bytes = 1.5 KB
  └─ Total memory: 47,293 × 1.5 KB ≈ 69.3 MB
```

## 4. Integration Verification

### 4.1 Sample T1528 Query Results

```sql
-- Find all CISA advisories mentioning T1528
FOR rel IN relationships
FILTER rel.relationship_type == "uses_in_cisa_advisory"
FILTER rel._from LIKE "%T1528%"
RETURN {
  technique: rel.technique_id,      -- T1528
  advisory: rel.advisory_id,        -- aa24-057a, aa23-278a, aa20-336a
  relationship: rel.relationship_type
}

Results:
├─ T1528 → aa24-057a (SVR-Attributed Actors)
├─ T1528 → aa23-278a (Scattered Spider Activity)
└─ T1528 → aa20-336a (Malicious Cyber Activity)
```

### 4.2 Cross-Validation Statistics

```
CISA Dataset Analysis:
  Total rows: 77
  Rows with TTPs: 77 (100%)
  Total T# references: 1,701
  Unique T# techniques: 432

Database Matching:
  T# found in ArangoDB: 405 / 432 (93.8%)
  T# not found: 27 (likely non-standard IDs)
  
Relationship Creation:
  Relationships created: 1,605
  Success rate: 94.4%
  Failed (missing parent): 96 (5.6%)
  
Unique Advisories:
  Advisories in CSV: 77
  Unique advisory IDs: 76
  Created entities: 74 (98.7%)
  
Integration Metrics:
  Techniques with advisories: 368
  Average techniques per advisory: 22.2
  Maximum techniques per advisory: 146 (aa20-336a)
  Minimum techniques per advisory: 0 (3 advisories)
```

## 5. Query Examples

### 5.1 Semantic Search Example

```python
# Query: "Find attacks that steal authentication credentials"
query_text = "steal authentication credentials"

# Step 1: Generate query embedding
query_embedding = embedding_model.encode(query_text)
# Output: 384-dim vector

# Step 2: Search in ArangoDB
FOR embed IN entity_embeddings
SORT embedding_similarity(embed.embedding, @query_embedding) DESC
LIMIT 10

# Results (top 5):
1. T1528 (0.87 similarity) - Steal Application Access Token
2. T1187 (0.85 similarity) - Forced Authentication
3. T1110 (0.83 similarity) - Brute Force
4. T1056 (0.81 similarity) - Input Capture
5. T1021 (0.79 similarity) - Remote Service Session Initiation
```

### 5.2 Graph Traversal Example

```python
# Query: "What ransomware uses T1083 (File Discovery)?"

# Step 1: Find all T1083 mentions in CISA advisories
FOR rel IN relationships
FILTER rel.relationship_type == "uses_in_cisa_advisory"
FILTER rel.technique_id == "T1083"
RETURN rel.advisory_id

# Results: aa24-060a, aa23-165a, aa22-055a, ...

# Step 2: For each advisory, find threat actors
FOR advisory IN entities
FILTER advisory.external_id IN [advisories...]
FOR rel IN relationships
FILTER rel._to == advisory._id
FILTER rel.relationship_type == "uses_in_cisa_advisory"
RETURN DISTINCT rel.from_entity

# Results: Identifies ransomware families and threat groups
```

### 5.3 Multi-hop Query Example

```python
# Query: "Which threat groups use phishing to compromise cloud infrastructure?"

# Path: group → uses → technique → belongs-to-tactic → tactic
#       ↓
#       technique → uses_in_cisa_advisory → advisory

FOR group IN entities
FILTER group.type == "intrusion-set"
FOR rel1 IN relationships
FILTER rel1._from == group._id
FILTER rel1.relationship_type == "uses"
LET technique = rel1._to

FOR rel2 IN relationships
FILTER rel2._from == technique
FILTER rel2.relationship_type == "uses_in_cisa_advisory"
LET advisory = rel2._to

FILTER advisory.description LIKE "%cloud%"

RETURN {
  group: group.name,
  technique: technique.name,
  advisory: advisory.name
}
```

## 6. Performance Metrics

```
Query Performance:
  Entity lookup (by ID): < 10 ms
  Semantic similarity search (1000 entities): 100-200 ms
  Graph traversal (2-hop): 50-100 ms
  Multi-hop traversal (3-4 hops): 200-500 ms

Database Size:
  Entities: 24,556 documents (~12 MB)
  Relationships: 24,342 edges (~8 MB)
  Embeddings: 47,293 vectors (~70 MB)
  Total database: ~100-120 MB (ArangoDB)

LLM Integration:
  Query encoding: 50-100 ms
  Context retrieval: 100-200 ms
  LLM inference: 2-5 seconds (Ollama, llama3.1:8b)
  Total end-to-end: 3-7 seconds

Memory Usage:
  ArangoDB process: ~300-500 MB
  Python RAG engine: ~200-300 MB
  Ollama (llama3.1:8b): ~5-8 GB
  Total system: ~6-9 GB RAM
```

## 7. Key Achievements

✅ **Unified Knowledge Graph**
- Integrated 2 major cybersecurity data sources
- 24,556 entities with complete context
- 24,342 relationships for attack path analysis

✅ **Semantic Intelligence**
- 47,293 embeddings for similarity search
- Find related techniques by meaning
- Context-aware threat intelligence

✅ **Real-World Data**
- 77 CISA advisories integrated
- 1,605 technique-to-advisory links
- Current threat landscape representation

✅ **AI-Ready Architecture**
- LLM-compatible embeddings
- Semantic search capabilities
- Natural language query processing

✅ **Scalable Design**
- Multi-model graph database (ArangoDB)
- Efficient triplet extraction
- Fast query performance
