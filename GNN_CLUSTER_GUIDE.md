# GNN Cluster Assignments & Boundary Analysis

## Overview

This document describes the cluster assignments for all 823 MITRE ATT&CK techniques based on GNN embeddings, including boundary annotations that identify which techniques are core cluster members vs. potential bridges between clusters.

## Files Generated

### 1. **gnn_cluster_assignments.tsv** (824 rows)
Complete assignment of every technique to a cluster with detailed distance metrics.

**Columns:**
- `index`: Technique index (0-822)
- `id`: MITRE technique ID (e.g., T1055.011)
- `name`: Full technique name
- `cluster`: Cluster ID (0, 1, or 2)
- `cluster_name`: Descriptive cluster name
- `distance_to_centroid`: Distance from technique to its assigned cluster center
- `distance_to_nearest_other_cluster`: Distance to nearest other cluster's center
- `boundary_ratio`: Ratio of distance to nearest other cluster / distance to own cluster
- `boundary_position`: Core, Interior, or Boundary

**Example:**
```
T1047   Windows Management Instrumentation   Cluster 2   distance: 8.49   boundary_ratio: 1.08   Boundary
```

---

### 2. **gnn_cluster_boundaries.tsv** (4 rows)
Cluster-level boundary statistics and geometry.

**Key Metrics:**
- `centroid_x`, `centroid_y`, `centroid_z`: Position of cluster center in 64-dim space
- `radius_mean`: Average distance of techniques from cluster center
- `radius_std`: Standard deviation of distances
- `radius_min`: Closest technique to center
- `radius_max`: Farthest technique from center
- `core_radius`: P33 (inner third of cluster)
- `boundary_radius`: P95 (outer region where boundary techniques exist)

**Example:**
```
Cluster 0 (Evasion_Persistence):
  Centroid: (-0.20, 0.22, -0.36)
  Radius: 7.72 ± 1.53
  Core (p33): 6.93
  Boundary (p95): 10.75
  Techniques: 108 total (0 core, 25 interior, 83 boundary)
```

---

### 3. **gnn_cluster_boundary_techniques.tsv** (531 rows)
Techniques positioned on cluster boundaries - **these are the bridges between clusters**.

**Important:** These 530 techniques (64% of all techniques) are close to other clusters and could bridge attack chains across cluster boundaries.

**Columns:**
- `index`, `id`, `name`: Technique identification
- `cluster`: Primary cluster assignment
- `cluster_name`: Cluster name
- `boundary_ratio`: How close they are to other clusters (1.0 = equally far, >2.0 = strongly in one cluster)

**Example (Bridge Techniques):**
```
T1047   Windows Management Instrumentation   Cluster: 2 (Discovery)   Ratio: 1.08
T1060   Registry Run Keys                     Cluster: 0 (Evasion)     Ratio: 1.00
T1558   Ccache Files                          Cluster: 0 (Evasion)     Ratio: 1.01
```

**Interpretation:**
- Ratio ~1.0 = **Strong Bridge** - equally close to multiple clusters
- Ratio 1.5-2.0 = **Medium Bridge** - leans toward one cluster but connects to others
- Ratio >2.0 = **Weak Bridge** - firmly in one cluster

---

### 4. **gnn_cluster_interior_techniques.tsv** (293 rows)
Techniques deep within clusters, representative of cluster characteristics.

**Columns:**
- `index`, `id`, `name`: Technique identification
- `cluster`, `cluster_name`: Cluster assignment
- `distance_to_centroid`: Distance from cluster center (lower = more central)

**Example (Core Cluster Members):**
```
Cluster 0:  T1189 Drive-by Compromise (distance: 5.42)
Cluster 1:  T1055.011 Extra Window Memory Injection (distance: 7.54)
Cluster 2:  T1589 Gather Victim Host Information (distance: 6.77)
```

---

### 5. **gnn_cluster_visualization.tsv** (824 rows)
Projection data for creating boundary visualizations.

**Columns:**
- All assignment columns plus:
- `dim1`, `dim2`: First two embedding dimensions (for 2D plotting)
- Shows which techniques are interior vs. boundary in embedding space

---

### 6. **gnn_cluster_core_techniques.tsv** (1 row)
Currently empty - indicates no techniques are extremely deep in clusters (good coverage of spectrum).

---

## Cluster Definitions

### **Cluster 0: Evasion & Persistence** 🔴
- **Size:** 108 techniques (13.1%)
- **Centroid:** (-0.20, 0.22)
- **Boundary Profile:**
  - Core (p33): 0 techniques
  - Interior: 25 techniques (23%)
  - Boundary: 83 techniques (77%)

**Characteristics:**
- Low-level persistence mechanisms (bootkit, UEFI, firmware)
- Anti-forensics and indicator removal
- Malware configuration and obfuscation

**Key Techniques:**
```
T1027   Obfuscated Files (5 variants)
T1070   Indicator Removal (4 variants)
T1542   Pre-OS Boot (3 variants)
T1552   Unsecured Credentials (3 variants)
T1561   Disk Wipe
T1133   External Remote Services
```

**Bridge Potential:** Very high (77% boundary) - these techniques bridge to both Cluster 1 (execution) and Cluster 2 (reconnaissance through credential exposure)

---

### **Cluster 1: Mixed General-Purpose** 🟢
- **Size:** 445 techniques (54.1%)
- **Centroid:** (0.12, -0.32)
- **Boundary Profile:**
  - Core: 0 techniques
  - Interior: 187 techniques (42%)
  - Boundary: 258 techniques (58%)

**Characteristics:**
- Versatile techniques used across multiple attack phases
- Process injection and execution methods
- Tool abuse and defense evasion
- Most techniques in this cluster are multi-purpose

**Key Techniques:**
```
T1546   Event Triggered Execution (13 variants)
T1564   Hide Artifacts (11 variants)
T1574   Hijack Execution Flow (11 variants)
T1218   Abuse System Tools (11 variants)
T1047   Windows Management Instrumentation
T1055   Process Injection (8 variants)
```

**Bridge Potential:** Moderate-high (58% boundary) - these bridge between evasion and discovery

---

### **Cluster 2: Discovery & Reconnaissance** 🔵
- **Size:** 270 techniques (32.8%)
- **Centroid:** (-0.12, 0.44)
- **Boundary Profile:**
  - Core: 0 techniques
  - Interior: 81 techniques (30%)
  - Boundary: 189 techniques (70%)

**Characteristics:**
- Initial access and reconnaissance
- Credential acquisition and brute force
- Information gathering and threat actor infrastructure
- Phishing and social engineering

**Key Techniques:**
```
T1003   OS Credential Dumping (7 variants)
T1588   Obtain Capabilities (6 variants)
T1566   Phishing (5 variants)
T1110   Brute Force (4 variants)
T1087   Account Discovery (4 variants)
```

**Bridge Potential:** Very high (70% boundary) - bridges between reconnaissance and execution/persistence

---

## Boundary Position Interpretation

### **Interior Techniques** (293 total)
- **Characteristic:** 30-42% of each cluster
- **Meaning:** Deeply embedded in cluster, representative of cluster's core function
- **Use Case:** When you want pure cluster behavior, use interior techniques
- **Example:** T1189 (Drive-by Compromise) is interior to Evasion cluster

### **Boundary Techniques** (530 total)
- **Characteristic:** 58-77% of each cluster
- **Meaning:** On cluster edge, likely bridging to other clusters
- **Use Case:** For understanding attack chains and transitions between phases
- **Ratio Interpretation:**
  - **1.0-1.2:** Strong bridge - equidistant from multiple clusters
  - **1.2-2.0:** Moderate bridge - leans toward primary cluster
  - **>2.0:** Weak bridge - committed to primary cluster but still boundary

**Example Bridge Sequence (Attack Chain):**
```
Reconnaissance (Cluster 2 Interior)
    ↓ [Bridge: T1589 - Gather Info, ratio 1.04]
    ↓
Credential Theft (Cluster 2 Boundary, ratio 1.10)
    ↓ [Bridge: T1078 - Valid Accounts, ratio 1.15]
    ↓
Lateral Movement (Cluster 1 Boundary, ratio 1.22)
    ↓ [Bridge: T1055 - Process Injection, ratio 1.23]
    ↓
Persistence (Cluster 0 Interior, ratio 1.40)
```

---

## Querying the Data

### Find all boundary techniques in Cluster 0:
```bash
grep "^[^0-9]*0[^0-9]" gnn_cluster_assignments.tsv | grep "Boundary"
```

### Find techniques closest to cluster center:
```bash
sort -k6 -n gnn_cluster_assignments.tsv | head -20
```

### Find strongest bridges (ratio ≈ 1.0):
```bash
awk -F'\t' '$7 < 1.05 && $7 > 0.95' gnn_cluster_assignments.tsv | head -20
```

### Get all techniques in a specific cluster:
```bash
grep "^.*Cluster 1" gnn_cluster_assignments.tsv | wc -l
```

---

## Statistical Summary

| Metric | Value |
|--------|-------|
| Total Techniques | 823 |
| Core Techniques | 0 |
| Interior Techniques | 293 (35.6%) |
| Boundary Techniques | 530 (64.4%) |
| Avg Cluster Size | 274 |
| Std Dev Cluster Size | 168 |
| Avg Distance to Own Centroid | 7.41 |
| Avg Distance to Nearest Other | 7.77 |

---

## Use Cases

### **1. Attack Chain Analysis**
Use boundary techniques to trace realistic attack progression:
- Start with Cluster 2 boundary (reconnaissance bridge)
- Transition through bridges to Cluster 1 (execution)
- End in Cluster 0 interior (persistence)

### **2. Detection Rule Development**
- **High Confidence:** Focus on interior techniques (more homogeneous)
- **Broader Coverage:** Use boundary techniques (catches variants and hybrids)

### **3. Clustering Validation**
Compare your cluster assignments with official MITRE ATT&CK tactic groups:
- **Expected:** Some mismatches due to multi-tactic techniques
- **Good:** Cluster 2 should align with Reconnaissance tactic
- **Good:** Cluster 0 should align with Persistence/Defense Evasion
- **Expected:** Cluster 1 mixed with Execution, Lateral Movement, Collection

### **4. Threat Actor Pattern Analysis**
Analyze which techniques specific threat actors prefer:
- Do they use boundary techniques (versatile, multi-tactic)?
- Or interior techniques (specialized, single-purpose)?

### **5. Anomaly Detection**
- Boundary ratio < 1.1 = unusual technique (might be novel/unknown)
- Boundary ratio > 2.5 = normal, well-understood technique
- Missing from any cluster = potential data quality issue

---

## Next Steps

1. **Visualize Boundaries:** Use gnn_cluster_visualization.tsv to plot boundaries
2. **Trace Attack Chains:** Connect bridge techniques across clusters
3. **Compare with RAG:** Compare cluster assignments with original 384-dim RAG embeddings
4. **Threat Modeling:** Map threat actors to cluster preferences
5. **Validate:** Cross-reference with official MITRE ATT&CK tactic assignments

---

## Technical Notes

- **Clustering Method:** K-means with k=3, initialized with k-means++
- **Distance Metric:** Euclidean distance in 64-dimensional GNN embedding space
- **Cluster Quality:** Silhouette Score 0.0621 (positive, indicating cluster structure exists)
- **Davies-Bouldin Index:** 3.41 (moderate overlap, realistic for cybersecurity domain)
- **Comparison with RAG:** ARI 0.67 (67% agreement with original embeddings)

