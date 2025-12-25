
================================================================================
MITRE2KG GRAPH DATABASE: SCHEMA & ARCHITECTURE SUMMARY
================================================================================

PROJECT OVERVIEW
================

The MITRE2kg graph database integrates two critical cybersecurity data sources:

1. MITRE ATT&CK Enterprise Framework (enterprise_attack.json)
   - Standardized knowledge base of adversary tactics and techniques
   - 14,000+ mappings across technologies

2. CISA Cybersecurity Advisories (CISA-crawl-rt-ttp-ct.csv)
   - 77 real-world cybersecurity advisories
   - 432 unique attack techniques (T#)
   - Sector-specific threat intelligence

This integration creates a unified knowledge graph enabling:
- Semantic search across attack techniques and advisories
- Threat actor attribution and campaign tracking
- Incident response guidance based on real vulnerabilities
- AI-powered analysis via Ollama LLM integration

================================================================================
DATABASE SCHEMA
================================================================================

DATABASE NAME: MITRE2kg
Location: ArangoDB @ http://localhost:8529
Credentials: root/openSesame

COLLECTIONS
============

1. entities (Primary Entity Collection)
   - Type: Document collection
   - Total documents: 24,556
   
   Entity Types:
   - attack-pattern: 823
     (Core MITRE techniques: parent + subtechniques)
   - tactic: 0
     (MITRE ATT&CK tactics: Recon, Initial Access, Execution, etc.)
   - group: 0
     (Threat actors and adversary groups)
   - software: 0
     (Tools, malware, and utilities)
   - malware: 695
   - campaign: 52
   - intrusion-set: 178
   - cisa_advisory: 74
     (CISA cybersecurity advisories from crawl dataset)

2. relationships (Edge Collection)
   - Type: Directed edge collection
   - Total edges: 24,342
   
   Relationship Types:
   - subtechnique-of: 476
     (T1234.001 subtechnique-of T1234 parent)
   - mitigates: 1,445
     (Countermeasures and mitigations)
   - detects: 691
     (Detection mechanisms)
   - uses: 17,270
     (Threat actors use techniques)
   - belongs-to-tactic: 1,071
     (Technique belongs to tactic)
   - attributed-to: 25
     (Attribution to threat actors)
   - revoked-by: 140
     (Technique revoked and replaced)
   - referenced_in_cisa_advisory: 1,619
     (Advisory mentions technique)
   - uses_in_cisa_advisory: 1,605
     (Technique used in attack described by advisory)

3. entity_embeddings (Semantic Vector Collection)
   - Type: Document collection
   - Documents: 24,556
   - Embedding model: all-MiniLM-L6-v2
   - Vector dimensions: 384
   - Stores: Dense semantic vectors for every entity
   - Used for: Similarity search, semantic queries

4. relationship_embeddings (Semantic Vector Collection)
   - Type: Document collection
   - Documents: 22,737
   - Embedding model: all-MiniLM-L6-v2
   - Vector dimensions: 384
   - Stores: Dense semantic vectors for relationships
   - Used for: Relationship similarity, context retrieval

================================================================================
DATA GENERATION & TRIPLET EXTRACTION
================================================================================

SOURCE 1: MITRE Enterprise Attack Framework (enterprise_attack.json)
============================================================================

Process:
1. Parse enterprise_attack.json (STIX 2.1 JSON format)
2. Extract entity objects:
   - attack-pattern (techniques)
   - tactic (attack phases)
   - group (threat actors)
   - software (malware/tools)
   - relationship (entity connections)

3. Normalize external IDs:
   - Attack patterns: T1234 (parent), T1234.001 (subtechnique)
   - Tactics: TA0001, TA0002, etc.
   - Groups: G0001, G0002, etc.

4. Extract relationships (triplets):
   - (attack-pattern, subtechnique-of, attack-pattern)
   - (attack-pattern, belongs-to-tactic, tactic)
   - (group, uses, attack-pattern)
   - (attack-pattern, mitigates, attack-pattern)
   - (attack-pattern, detects, attack-pattern)

5. Generate embeddings:
   - Use attack pattern name + description → all-MiniLM-L6-v2
   - Create 384-dimensional vectors
   - Store in entity_embeddings collection
   - L2-normalize all vectors

Result: 823 attack patterns with complete relationship graph


SOURCE 2: CISA Cybersecurity Advisories (CISA-crawl-rt-ttp-ct.csv)
============================================================================

Data Source: https://www.cisa.gov/news-events/cybersecurity-advisories
Columns:
  - RawText: Full advisory text
  - CleanText: Cleaned advisory content
  - TTP: MITRE T# techniques (comma/newline separated)
  - URL: Advisory URL (contains advisory ID: aa24-060a, etc.)

Processing Steps:

1. Load CSV (77 advisories)
   - Parse 77 rows from CISA crawl dataset
   - Extract advisory ID from URL (aa24-060a format)
   - Extract T# technique IDs from TTP field (T1083, T1071.002, etc.)

2. Create CISA advisory entities:
   - For each unique advisory ID, create cisa_advisory entity
   - Store advisory metadata (ID, name, URL, original text)
   - Result: 74 unique CISA advisory entities

3. Create technique-to-advisory links:
   - For each TTP in advisory: create uses_in_cisa_advisory relationship
   - Link attack-pattern entity → CISA advisory entity
   - Indicates: "This technique is used in attacks described by this advisory"
   - Result: 1,605 relationships created (94.4% success rate)

4. Update embeddings:
   - Regenerate embeddings for all 823 techniques
   - Context now includes CISA advisory references
   - Enables semantic search: "Find techniques used in ransomware attacks"

5. Verify integration:
   - Cross-check TTP extraction accuracy
   - Validate relationship creation
   - Match records against ArangoDB entities


TRIPLET GENERATION PROCESS
============================================================================

Triplets (Subject → Relationship → Object) created via:

1. MITRE Framework (from enterprise_attack.json):
   - Subject: attack-pattern (T1055)
   - Relationship: subtechnique-of
   - Object: attack-pattern parent (T1055.004)
   
   - Subject: group (G0001)
   - Relationship: uses
   - Object: attack-pattern (T1234)
   
   - Subject: attack-pattern (T1234)
   - Relationship: belongs-to-tactic
   - Object: tactic (TA0001)

2. CISA Integration (from CISA-crawl CSV):
   - Subject: attack-pattern (T1083)
   - Relationship: uses_in_cisa_advisory
   - Object: cisa_advisory (aa24-060a)
   
   - Subject: attack-pattern (T1071.002)
   - Relationship: referenced_in_cisa_advisory
   - Object: cisa_advisory (aa24-038a)


STATISTICS
===========

Total Triplets (Edges): 24,342

Breakdown by relationship type:
  • attributed-to: 25
  • belongs-to-tactic: 1,071
  • detects: 691
  • mitigates: 1,445
  • referenced_in_cisa_advisory: 1,619
  • revoked-by: 140
  • subtechnique-of: 476
  • uses: 17,270
  • uses_in_cisa_advisory: 1,605


================================================================================
FINAL STATISTICS & METRICS
================================================================================

ENTITIES: 24,556
  └─ Attack patterns (parent + sub): 823
  └─ Tactics: 0
  └─ Threat groups: 0
  └─ Software/malware: 0 + 695
  └─ CISA advisories: 74

RELATIONSHIPS: 24,342
  └─ MITRE framework: 22,737
  └─ CISA integration: 1,605

EMBEDDINGS: 47,293 vectors
  └─ Entity embeddings: 24,556 (384-dim)
  └─ Relationship embeddings: 22,737 (384-dim)
  └─ Total memory: ~69.3 MB

GRAPH DENSITY
  └─ Average connections per entity: 0.99
  └─ Techniques with advisories: 823
  └─ Advisory coverage: 1,605 technique references

================================================================================
TECHNICAL ARCHITECTURE
================================================================================

Tech Stack:
  • Database: ArangoDB 3.x (Multi-model graph DB)
  • Python Libraries:
    - python-arango: Database queries (AQL)
    - sentence-transformers: Embeddings (all-MiniLM-L6-v2)
    - requests: HTTP for Ollama LLM
    - pandas: CSV processing
  • LLM: Ollama (llama3.1:8b running in Docker)
  • RAG Framework: Custom ArangoDB RAG Query Engine

Key Features:
  ✓ Semantic search across 384-dimensional embedding space
  ✓ Multi-hop graph traversal (find related techniques)
  ✓ LLM-powered question answering with graph context
  ✓ Real-world advisory integration
  ✓ End-to-end threat intelligence pipeline

================================================================================
