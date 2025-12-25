#!/usr/bin/env python3
"""
Analyze GNN Embedding Clusters
Extracts and characterizes the three clusters discovered by t-SNE visualization
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load GNN embeddings and metadata"""
    print("Loading GNN embeddings and metadata...")
    
    # Load embeddings
    embeddings = np.loadtxt('/home/vasanthiyer-gpu/gnn_embeddings_vectors.tsv')
    
    # Load metadata
    metadata = []
    with open('/home/vasanthiyer-gpu/gnn_embeddings_metadata.tsv', 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                metadata.append({
                    'index': int(parts[0]),
                    'id': parts[1],
                    'name': parts[2]
                })
    
    df = pd.DataFrame(metadata)
    print(f"Loaded {len(embeddings)} embeddings and {len(df)} metadata entries")
    return embeddings, df

def perform_clustering(embeddings, n_clusters=3):
    """Perform K-means clustering on embeddings"""
    print(f"\nPerforming K-means clustering with k={n_clusters}...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Calculate metrics
    silhouette = silhouette_score(embeddings, clusters)
    davies_bouldin = davies_bouldin_score(embeddings, clusters)
    
    print(f"Silhouette Score: {silhouette:.4f} (range: -1 to 1, higher is better)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
    
    return clusters, kmeans

def analyze_clusters(embeddings, clusters, df):
    """Analyze cluster characteristics"""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS")
    print("="*80)
    
    for cluster_id in range(3):
        indices = np.where(clusters == cluster_id)[0]
        cluster_techs = df[df['index'].isin(indices)].sort_values('index')
        
        print(f"\n{'─'*80}")
        print(f"CLUSTER {cluster_id + 1}")
        print(f"{'─'*80}")
        print(f"Size: {len(cluster_techs)} techniques ({len(cluster_techs)/len(df)*100:.1f}%)")
        
        # Analyze technique types from IDs
        tactic_counts = {}
        for tech_id in cluster_techs['id']:
            # Extract main tactic from ID (T1055 = 1055)
            main_id = tech_id.split('.')[0]
            if main_id not in tactic_counts:
                tactic_counts[main_id] = 0
            tactic_counts[main_id] += 1
        
        print(f"\nTop Technique IDs in cluster:")
        for tech_id, count in sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {tech_id}: {count} variants")
        
        print(f"\nFirst 20 techniques:")
        for idx, row in cluster_techs.head(20).iterrows():
            print(f"  {row['id']:15} - {row['name']}")
        
        # Calculate cluster statistics
        cluster_embeddings = embeddings[indices]
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        
        print(f"\nCluster Statistics:")
        print(f"  Mean distance to center: {np.mean(distances):.4f}")
        print(f"  Std dev of distances: {np.std(distances):.4f}")
        print(f"  Max distance from center: {np.max(distances):.4f}")
        print(f"  Median distance from center: {np.median(distances):.4f}")

def identify_bridges(embeddings, clusters, df, threshold=0.8):
    """Identify techniques that bridge clusters (likely in lateral movement)"""
    print("\n" + "="*80)
    print("BRIDGE ANALYSIS - Cross-Cluster Connections")
    print("="*80)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    # Find techniques close to cluster boundaries
    bridges = []
    for idx, (emb, cluster) in enumerate(zip(embeddings, clusters)):
        distances_to_centers = np.linalg.norm(kmeans.cluster_centers_ - emb, axis=1)
        sorted_clusters = np.argsort(distances_to_centers)
        
        # Bridge if closest to different cluster than assigned
        closest_cluster = sorted_clusters[0]
        if closest_cluster != cluster:
            tech = df[df['index'] == idx].iloc[0]
            bridges.append({
                'index': idx,
                'id': tech['id'],
                'name': tech['name'],
                'assigned_cluster': cluster,
                'nearest_cluster': closest_cluster,
                'distance_ratio': distances_to_centers[closest_cluster] / distances_to_centers[cluster]
            })
    
    bridges = sorted(bridges, key=lambda x: x['distance_ratio'])[:30]
    
    print(f"\nTop 30 Bridge Techniques (transitioning between clusters):\n")
    print(f"{'ID':<15} {'Name':<40} {'From→To':<15} {'Distance Ratio':<15}")
    print(f"{'-'*85}")
    
    for bridge in bridges:
        name = bridge['name'][:38]
        direction = f"C{bridge['assigned_cluster']}→C{bridge['nearest_cluster']}"
        print(f"{bridge['id']:<15} {name:<40} {direction:<15} {bridge['distance_ratio']:.4f}")

def compare_with_rag():
    """Compare GNN clustering with original RAG embeddings"""
    print("\n" + "="*80)
    print("COMPARISON: GNN vs RAG Embeddings")
    print("="*80)
    
    try:
        rag_embeddings = np.loadtxt('/home/vasanthiyer-gpu/embeddings_vectors.tsv')
        print(f"\nRAG Embeddings Shape: {rag_embeddings.shape} (384-dimensional)")
        print(f"GNN Embeddings Shape: {embeddings.shape} (64-dimensional)")
        
        # Cluster RAG embeddings
        rag_clusters, _ = perform_clustering(rag_embeddings, n_clusters=3)
        
        # Compare cluster assignments
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(clusters, rag_clusters)
        nmi = normalized_mutual_info_score(clusters, rag_clusters)
        
        print(f"\nAdjusted Rand Index: {ari:.4f} (0=independent, 1=identical)")
        print(f"Normalized Mutual Information: {nmi:.4f} (0=independent, 1=identical)")
        print(f"\nInterpretation:")
        if ari > 0.5:
            print("  ✓ GNN and RAG have similar cluster structure (good agreement)")
        else:
            print("  ℹ GNN discovered different structure than RAG (capturing graph relationships)")
            
    except FileNotFoundError:
        print("RAG embeddings not found - skipping comparison")

if __name__ == '__main__':
    # Load data
    embeddings, df = load_data()
    
    # Perform clustering
    clusters, kmeans = perform_clustering(embeddings, n_clusters=3)
    
    # Analyze clusters
    analyze_clusters(embeddings, clusters, df)
    
    # Identify bridges
    identify_bridges(embeddings, clusters, df)
    
    # Compare with RAG
    compare_with_rag()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
