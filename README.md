# MITRE GNN GraphRAG Analysis

A comprehensive analysis framework integrating MITRE ATT&CK techniques with Graph Neural Networks (GNN) and GraphRAG for advanced threat intelligence and knowledge graph analysis.

## 📋 Project Overview

This project combines three powerful approaches for cybersecurity analysis:

- **MITRE ATT&CK**: Adversary tactics and techniques knowledge base
- **Graph Neural Networks (GNN)**: Neural network models on graph-structured data
- **GraphRAG**: Graph-based Retrieval-Augmented Generation for knowledge retrieval

## 🎯 Key Features

- **GNN Clustering**: Advanced clustering of MITRE techniques using graph neural networks
- **Embeddings & Visualization**: 2D and 3D embedding visualization of technique relationships
- **GraphRAG Integration**: RAG-based queries over MITRE knowledge graphs
- **Evaluation Framework**: Comprehensive evaluation tools for RAG vs Graph comparison
- **LLM Judge**: AI-powered evaluation of retrieval quality
- **Automated Metrics**: Batch evaluation with detailed reporting

## 📁 Project Structure

```
├── Python Scripts
│   ├── create_gnn_tsv_embeddings.py      # Generate GNN embeddings
│   ├── analyze_gnn_clusters.py           # Analyze GNN cluster results
│   ├── mitre_graphrag_gnn.py             # Main GraphRAG integration
│   ├── mitre_llm_judge.py                # LLM-based evaluation
│   ├── mitre_automated_metrics.py        # Automated metric calculation
│   └── [other analysis scripts]
│
├── Data Files
│   ├── embeddings_vectors.tsv            # GNN embeddings
│   ├── gnn_embeddings_*.tsv              # Clustered embeddings
│   ├── mitre_embeddings.*                # MITRE embeddings (CSV/JSON)
│   └── [visualization & cluster data]
│
├── Documentation
│   ├── START_HERE.md                     # Getting started guide
│   ├── GRAPHRAG_GNN_GUIDE.md             # GraphRAG setup guide
│   ├── GNN_CLUSTER_GUIDE.md              # GNN clustering guide
│   ├── GNN_EMBEDDINGS_GUIDE.md           # Embeddings guide
│   ├── EVALUATION_TOOLKIT_README.md      # Evaluation framework
│   └── [detailed guides & reports]
│
├── Visualization
│   ├── gnn_embeddings_viewer.html        # Interactive 3D viewer
│   ├── evaluation_results.html           # Evaluation dashboard
│   └── [visualization assets]
│
└── Configuration & Utilities
    ├── COMMANDS_CHEATSHEET.sh            # Common commands
    ├── installer.sh                      # Setup script
    └── health_check.py                   # System health check
```

## 🚀 Getting Started

1. **Review the Documentation**
   ```bash
   cat START_HERE.md
   ```

2. **Check System Requirements**
   ```bash
   python health_check.py
   ```

3. **Install Dependencies**
   ```bash
   bash installer.sh
   ```

4. **Run Analysis**
   ```bash
   python mitre_graphrag_gnn.py
   ```

## � Required Datasets

This project requires two external datasets. Download them from the links below and place in a `datasets/` folder:

### 1. MITRE Enterprise ATT&CK Framework
- **File**: `enterprise-attack.json` (~44 MB)
- **Source**: [MITRE ATT&CK GitHub](https://github.com/mitre-attack/attack-stix-data/blob/master/enterprise-attack.json)
- **Description**: Complete MITRE ATT&CK framework with all tactics, techniques, and relationships
- **Usage**: Core knowledge base for technique analysis

### 2. CISA Crawled Real-Time TTP & CTs Dataset
- **File**: `CISA-crawl-rt-ttp-ct.csv`
- **Source**: [CISA ATT&CK Mappings](https://cisagov.github.io/mitre-attack-mappings/) or [CISA GitHub](https://github.com/cisagov)
- **Description**: CISA's real-time crawled TTP (Tactics, Techniques, Procedures) and Control Techniques mappings
- **Usage**: Threat intelligence and vulnerability correlation

**Setup Instructions**:
```bash
# Create datasets directory
mkdir -p datasets

# Download the files and place them in the datasets/ folder
# enterprise-attack.json → datasets/enterprise-attack.json
# CISA-crawl-rt-ttp-ct.csv → datasets/CISA-crawl-rt-ttp-ct.csv

# Verify downloads
ls -lh datasets/
```

## 📊 Generated Data Files

### Embeddings
- `embeddings_vectors.tsv` - GNN-generated embeddings
- `mitre_embeddings.json` - Formatted MITRE embeddings
- `gnn_embeddings_2d.tsv` - 2D projection for visualization
- `gnn_embeddings_3d.tsv` - 3D projection for visualization

### Clustering Results
- `gnn_cluster_assignments.tsv` - Technique-to-cluster mappings
- `gnn_cluster_boundaries.tsv` - Cluster boundary definitions
- `gnn_cluster_core_techniques.tsv` - Core techniques per cluster
- `gnn_cluster_interior_techniques.tsv` - Interior techniques per cluster

### Evaluation Results
- `evaluation_results.html` - Visual evaluation dashboard
- `batch_test_results.json` - Batch evaluation metrics
- `health_check_report.json` - System health status

## 🔍 Core Analysis Tools

### GNN Analysis
- **Clustering**: Unsupervised grouping of MITRE techniques
- **Embeddings**: Dense vector representations for similarity analysis
- **Visualization**: Interactive exploration of technique relationships

### GraphRAG Framework
- **Knowledge Graph Construction**: Build graphs from MITRE data
- **Retrieval-Augmented Generation**: Query-based technique retrieval
- **LLM Integration**: AI-powered knowledge synthesis

### Evaluation & Metrics
- **RAG vs Graph Comparison**: Benchmark different approaches
- **LLM Judge**: Semantic quality assessment
- **Automated Metrics**: Precision, recall, F1, and custom metrics

## 📈 Evaluation Framework

The project includes comprehensive evaluation tools:

- **Automated Metrics**: `mitre_automated_metrics.py`
- **LLM Judge**: `mitre_llm_judge.py`
- **RAG Comparison**: `mitre_rag_vs_graph_comparison.py`
- **Triple Evaluation**: `mitre_triple_evaluator.py`

See `EVALUATION_TOOLKIT_README.md` for detailed instructions.

## 🎨 Visualization

Interactive visualizations available:

- **3D Embeddings Viewer**: `gnn_embeddings_viewer.html` - Explore GNN embeddings in 3D
- **Evaluation Dashboard**: `evaluation_results.html` - View evaluation results
- **Cluster Visualization**: `gnn_cluster_visualization.tsv` - Cluster boundary data

## 📚 Documentation

Comprehensive guides available:

- `START_HERE.md` - Quick start guide
- `GRAPHRAG_GNN_GUIDE.md` - GraphRAG setup and usage
- `GNN_CLUSTER_GUIDE.md` - GNN clustering methodology
- `GNN_EMBEDDINGS_GUIDE.md` - Working with embeddings
- `RAG_VS_GRAPH_EVALUATION_GUIDE.md` - Evaluation framework
- `GRAPHDB_ARCHITECTURE.md` - Database schema and design
- `LLM_JUDGE_GUIDE.md` - Using the LLM judge

## 🛠️ Utilities

- `COMMANDS_CHEATSHEET.sh` - Common shell commands
- `health_check.py` - Verify system setup
- `quick_eval.py` - Quick evaluation runner
- `installer.sh` - Environment setup

## 📋 Reports & Summaries

- `FINAL_SUMMARY.txt` - Project completion summary
- `STATUS_REPORT.txt` - Current project status
- `FILE_INVENTORY.md` - Complete file listing
- `COMPARISON_SUMMARY.txt` - RAG vs Graph comparison results
- `SOLUTION_SUMMARY.md` - Solution overview

## 🔧 Requirements

- Python 3.8+
- Graph Neural Network libraries (PyTorch Geometric, etc.)
- GraphRAG framework
- LLM integration (OpenAI/Ollama)
- Data processing tools (pandas, numpy)

See `installer.sh` for automatic setup.

## 📞 Quick Reference

See `QUICK_REFERENCE.md` for common tasks and commands.

## 📄 License

This project integrates MITRE ATT&CK framework under appropriate licensing terms.

## 🤝 Contributing

For questions or improvements, refer to the documentation files or project guidelines.

---

**Last Updated**: December 24, 2025

**Project Status**: Active

**Repository**: https://github.com/viyer-research/mitre-gnn-analysis
