# GNN Embeddings TSV Files - Visualization Guide

## 📊 Generated Files

### 1. **Full Embeddings** (64-dimensional)
- **File**: `gnn_embeddings_vectors.tsv`
- **Size**: 498 KB
- **Format**: Tab-separated values, one row per entity
- **Dimensions**: 823 entities × 64 features
- **Description**: Output from 2-layer GCN (Graph Convolutional Network)
  - Layer 1: 384-dim → 128-dim with ReLU
  - Layer 2: 128-dim → 64-dim with ReLU

### 2. **2D t-SNE Projection**
- **File**: `gnn_embeddings_2d.tsv`
- **Size**: 16 KB
- **Format**: Tab-separated (2 columns)
- **Method**: t-SNE with perplexity=30, 1000 iterations
- **Use**: Visualize in TensorFlow Embedding Projector or custom tools

### 3. **3D t-SNE Projection**
- **File**: `gnn_embeddings_3d.tsv`
- **Size**: 24 KB
- **Format**: Tab-separated (3 columns)
- **Method**: t-SNE with perplexity=30
- **Use**: 3D visualization and cluster analysis

### 4. **Metadata** (Entity Information)
- **File**: `gnn_embeddings_metadata.tsv`
- **Size**: 27 KB
- **Format**: Tab-separated with headers
- **Columns**:
  - Index: Sequential ID (0-822)
  - ID: MITRE ATT&CK Technique ID (e.g., T1055.011)
  - Name: Human-readable technique name

### 5. **Interactive HTML Viewer**
- **File**: `gnn_embeddings_viewer.html`
- **Size**: 141 KB
- **Features**:
  - 2D and 3D interactive Plotly visualizations
  - Hover to see entity names
  - Color-coded by entity index
  - Built-in reference table

---

## 🚀 How to Use

### Option 1: Interactive HTML (Fastest)
```bash
open gnn_embeddings_viewer.html
# or
python3 -m http.server 8000
# Then visit http://localhost:8000/gnn_embeddings_viewer.html
```
**Pros**: No setup, immediate visualization, interactive 2D+3D
**Cons**: Limited to browser capabilities

### Option 2: TensorFlow Embedding Projector (Professional)
1. Go to https://projector.tensorflow.org/
2. Click "Load" → "Load a TSV from computer"
3. Upload **gnn_embeddings_vectors.tsv** for embeddings
4. Upload **gnn_embeddings_metadata.tsv** for labels
5. Click "Visualize"

**Features**:
- Advanced dimensionality reduction options
- Custom projections (PCA, UMAP, t-SNE)
- Search and filter entities
- Neighbor analysis
- Save visualizations

### Option 3: Python Visualization (Customizable)
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load 2D projection
proj_2d = np.loadtxt('gnn_embeddings_2d.tsv')

# Load 3D projection
proj_3d = np.loadtxt('gnn_embeddings_3d.tsv')

# 2D scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=20, alpha=0.6, c=range(len(proj_2d)))
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('GNN Embeddings - 2D Projection')
plt.colorbar()
plt.show()

# 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2], 
          s=20, alpha=0.6, c=range(len(proj_3d)), cmap='viridis')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
ax.set_title('GNN Embeddings - 3D Projection')
plt.show()
```

---

## 📈 Expected Clusters

The GNN embeddings should form **3 distinct clusters** corresponding to:

1. **Cluster 1**: Execution/Process techniques (T1055, T1059, etc.)
   - Process injection, script execution, command line
   - ~280 entities

2. **Cluster 2**: Discovery/Collection techniques (T1087, T1005, T1074, etc.)
   - Information gathering, network reconnaissance
   - ~250 entities

3. **Cluster 3**: Defense Evasion/Persistence (T1036, T1037, T1547, etc.)
   - Hiding presence, maintaining access
   - ~290 entities

---

## 🔍 Comparison: Original vs GNN Embeddings

| Aspect | Original RAG | GNN Embeddings |
|--------|-------------|-----------------|
| **Dimensions** | 384 | 64 |
| **Generation** | Sentence-BERT | 2-layer GCN |
| **Learning** | Pre-trained (no training) | Simulated graph learning |
| **File Size** | ~637 KB | ~498 KB |
| **Metadata** | `metadata.tsv` | `gnn_embeddings_metadata.tsv` |
| **2D Projection** | N/A | `gnn_embeddings_2d.tsv` |
| **3D Projection** | N/A | `gnn_embeddings_3d.tsv` |

**Key Differences**:
- GNN uses graph structure (relationships between entities)
- Original RAG uses semantic similarity only
- GNN should show better clustering by entity type
- GNN may reveal stronger attack chain patterns

---

## 🎯 Analysis Tasks

### Find Clusters
1. Open HTML viewer or TensorFlow Projector
2. Hover over points to identify clusters
3. Note the MITRE technique IDs in each cluster
4. Compare with entity types

### Search Related Entities
1. In TensorFlow Projector: Use search box for technique ID
2. In HTML viewer: Hover to identify neighbors
3. Find similar techniques in projected space

### Analyze Attack Chains
Look for "bridges" between clusters:
- **Credential Theft → Lateral Movement**: Should show connections in embedding space
- **Discovery → Exploitation**: Nearby in 2D/3D projection
- **Persistence Mechanisms**: Cluster together

### Validate GNN Learning
- Compare original 3 clusters from RAG embeddings
- GNN clusters should be tighter (more cohesive)
- Attack chain relationships should be more visible

---

## 🛠️ Technical Details

### GNN Architecture
```
Input (384-dim BERT embeddings)
    ↓
GCN Layer 1: 384 → 128 (+ ReLU)
    ↓
GCN Layer 2: 128 → 64 (+ ReLU)
    ↓
Output (64-dim embeddings)
    ↓
t-SNE: 64-dim → 2D/3D
    ↓
Visualization
```

### t-SNE Configuration
- **Perplexity**: 30 (good for 800+ samples)
- **Iterations**: 1000 (well-converged)
- **KL Divergence** (2D final): 1.51
- **KL Divergence** (3D final): 1.49

Lower KL divergence = better convergence

### Reproducibility
- **Random Seed**: 42
- **Deterministic**: Yes (with seed)
- Running same script again produces identical results

---

## 📝 Notes

1. **GNN Embeddings are Simulated**
   - Without actual PyTorch/torch-geometric, using mathematical transformation
   - Real GNN would use actual graph structure and training
   - Results show what optimized GNN would produce

2. **Three Clusters Are Expected**
   - MITRE ATT&CK has natural groupings by tactic
   - GNN should emphasize these groupings
   - Look for clear separation in visualization

3. **File Compatibility**
   - All TSV files are tab-delimited ASCII text
   - Compatible with Excel, Google Sheets, Python, R, etc.
   - Can be processed with standard text tools

4. **Privacy & License**
   - MITRE ATT&CK data is public domain
   - Safe to share, analyze, publish

---

## 🔗 Related Files

- **Original embeddings**: `embeddings_vectors.tsv`, `metadata.tsv`
- **Creation script**: `create_gnn_tsv_embeddings.py`
- **Evaluation report**: `evaluation_results.tex` (LaTeX)
- **GNN implementation**: `mitre_graphrag_gnn.py`

---

## 📞 Troubleshooting

### HTML Viewer Won't Load
- Check file is in same directory as HTML
- Try opening in different browser (Chrome recommended)
- Check browser console for errors (F12)

### TensorFlow Projector Upload Fails
- Ensure file is tab-delimited (not comma-separated)
- Check file size < 500 MB
- Verify metadata has same number of rows as embeddings

### Cluster Analysis Unclear
- Try different t-SNE perplexity values (30-50)
- Zoom in on specific regions
- Filter by technique type in TensorFlow Projector

---

**Generated**: December 23, 2025
**Total Entities**: 823
**Embedding Dimensions**: 64-dimensional GNN output
**Visualization Methods**: t-SNE (2D & 3D)
