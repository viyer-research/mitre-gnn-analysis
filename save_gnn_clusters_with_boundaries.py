#!/usr/bin/env python3
"""
Save GNN Cluster Assignments with Boundary Annotations
Creates detailed cluster files with distance metrics and boundary information
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
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
    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings, df

def perform_clustering_with_distances(embeddings, n_clusters=3):
    """Perform K-means and calculate distances to centroids"""
    print(f"Performing K-means clustering...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Calculate distance from each point to its cluster center
    distances_to_cluster = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        distances_to_cluster[i] = np.linalg.norm(embeddings[i] - kmeans.cluster_centers_[clusters[i]])
    
    # Calculate distance to nearest other cluster center
    distances_to_nearest_other = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        all_distances = np.linalg.norm(embeddings[i] - kmeans.cluster_centers_, axis=1)
        other_distances = np.concatenate([all_distances[:clusters[i]], all_distances[clusters[i]+1:]])
        distances_to_nearest_other[i] = np.min(other_distances)
    
    return kmeans, clusters, distances_to_cluster, distances_to_nearest_other

def save_cluster_assignments(df, clusters, distances_to_cluster, distances_to_nearest_other, kmeans):
    """Save cluster assignments with all distance metrics"""
    print("\nSaving cluster assignments...")
    
    # Create comprehensive dataframe
    cluster_df = pd.DataFrame({
        'index': df['index'],
        'id': df['id'],
        'name': df['name'],
        'cluster': clusters,
        'distance_to_centroid': distances_to_cluster,
        'distance_to_nearest_other_cluster': distances_to_nearest_other,
        'boundary_ratio': distances_to_nearest_other / (distances_to_cluster + 1e-6)
    })
    
    # Add cluster names
    cluster_names = {
        0: "Evasion_Persistence",
        1: "Mixed_General_Purpose",
        2: "Discovery_Reconnaissance"
    }
    cluster_df['cluster_name'] = cluster_df['cluster'].map(cluster_names)
    
    # Categorize by boundary position
    def categorize_boundary(ratio):
        if ratio > 2.0:
            return "Core"
        elif ratio > 1.2:
            return "Interior"
        else:
            return "Boundary"
    
    cluster_df['boundary_position'] = cluster_df['boundary_ratio'].apply(categorize_boundary)
    
    # Save full file
    output_file = '/home/vasanthiyer-gpu/gnn_cluster_assignments.tsv'
    cluster_df.to_csv(output_file, sep='\t', index=False)
    print(f"✓ Saved full assignments to: {output_file}")
    
    return cluster_df

def save_cluster_boundaries(cluster_df, kmeans, embeddings):
    """Save cluster boundary analysis"""
    print("\nCalculating cluster boundaries...")
    
    boundaries = []
    for cluster_id in range(3):
        cluster_mask = cluster_df['cluster'] == cluster_id
        cluster_indices = cluster_df[cluster_mask].index
        cluster_embeddings = embeddings[cluster_indices]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
        
        boundary_info = {
            'cluster_id': cluster_id,
            'cluster_name': cluster_df[cluster_mask]['cluster_name'].iloc[0],
            'num_techniques': len(cluster_indices),
            'centroid_x': cluster_center[0],
            'centroid_y': cluster_center[1],
            'centroid_z': cluster_center[2] if len(cluster_center) > 2 else 0,
            'radius_mean': float(np.mean(distances)),
            'radius_std': float(np.std(distances)),
            'radius_min': float(np.min(distances)),
            'radius_max': float(np.max(distances)),
            'radius_p95': float(np.percentile(distances, 95)),
            'core_radius': float(np.percentile(distances, 33)),  # Inner third
            'boundary_radius': float(np.percentile(distances, 95))  # Outer region
        }
        boundaries.append(boundary_info)
    
    boundaries_df = pd.DataFrame(boundaries)
    output_file = '/home/vasanthiyer-gpu/gnn_cluster_boundaries.tsv'
    boundaries_df.to_csv(output_file, sep='\t', index=False)
    print(f"✓ Saved boundary info to: {output_file}")
    
    return boundaries_df

def save_boundary_categorized(cluster_df):
    """Save techniques organized by their boundary position"""
    print("\nCategorizing by boundary position...")
    
    cluster_df_reset = cluster_df.reset_index(drop=True)
    
    # Core techniques (deeply embedded in cluster)
    core_techs = cluster_df_reset[cluster_df_reset['boundary_position'] == 'Core'][
        ['index', 'id', 'name', 'cluster', 'cluster_name', 'distance_to_centroid']
    ].sort_values(['cluster', 'distance_to_centroid'])
    
    core_file = '/home/vasanthiyer-gpu/gnn_cluster_core_techniques.tsv'
    core_techs.to_csv(core_file, sep='\t', index=False)
    print(f"✓ Saved {len(core_techs)} core techniques to: {core_file}")
    
    # Boundary techniques (on cluster edge, could bridge clusters)
    boundary_techs = cluster_df_reset[cluster_df_reset['boundary_position'] == 'Boundary'][
        ['index', 'id', 'name', 'cluster', 'cluster_name', 'boundary_ratio']
    ].sort_values(['cluster', 'boundary_ratio'])
    
    boundary_file = '/home/vasanthiyer-gpu/gnn_cluster_boundary_techniques.tsv'
    boundary_techs.to_csv(boundary_file, sep='\t', index=False)
    print(f"✓ Saved {len(boundary_techs)} boundary techniques to: {boundary_file}")
    
    # Interior techniques
    interior_techs = cluster_df_reset[cluster_df_reset['boundary_position'] == 'Interior'][
        ['index', 'id', 'name', 'cluster', 'cluster_name', 'distance_to_centroid']
    ].sort_values(['cluster', 'distance_to_centroid'])
    
    interior_file = '/home/vasanthiyer-gpu/gnn_cluster_interior_techniques.tsv'
    interior_techs.to_csv(interior_file, sep='\t', index=False)
    print(f"✓ Saved {len(interior_techs)} interior techniques to: {interior_file}")

def create_boundary_visualization(cluster_df, embeddings):
    """Create CSV files for visualization"""
    print("\nCreating visualization data...")
    
    # Project to 2D for visualization (first 2 dimensions)
    viz_data = pd.DataFrame({
        'index': cluster_df['index'],
        'id': cluster_df['id'],
        'name': cluster_df['name'],
        'cluster': cluster_df['cluster'],
        'cluster_name': cluster_df['cluster_name'],
        'boundary_position': cluster_df['boundary_position'],
        'dim1': embeddings[:, 0],
        'dim2': embeddings[:, 1],
        'distance_to_centroid': cluster_df['distance_to_centroid']
    })
    
    viz_file = '/home/vasanthiyer-gpu/gnn_cluster_visualization.tsv'
    viz_data.to_csv(viz_file, sep='\t', index=False)
    print(f"✓ Saved visualization data to: {viz_file}")

def print_summary(cluster_df, boundaries_df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("CLUSTER BOUNDARY SUMMARY")
    print("="*80)
    
    for _, row in boundaries_df.iterrows():
        cluster_id = int(row['cluster_id'])
        cluster_name = row['cluster_name']
        num_techs = int(row['num_techniques'])
        
        print(f"\n{cluster_name.upper()} (Cluster {cluster_id})")
        print(f"  Techniques: {num_techs}")
        print(f"  Centroid Position: ({row['centroid_x']:.2f}, {row['centroid_y']:.2f})")
        print(f"  Cluster Radius:")
        print(f"    • Mean: {row['radius_mean']:.2f}")
        print(f"    • Std Dev: {row['radius_std']:.2f}")
        print(f"    • Min (closest): {row['radius_min']:.2f}")
        print(f"    • Max (farthest): {row['radius_max']:.2f}")
        print(f"    • Core (p33): {row['core_radius']:.2f}")
        print(f"    • Boundary (p95): {row['boundary_radius']:.2f}")
        
        # Show distribution
        cluster_mask = cluster_df['cluster'] == cluster_id
        positions = cluster_df[cluster_mask]['boundary_position'].value_counts()
        print(f"  Technique Distribution:")
        for pos in ['Core', 'Interior', 'Boundary']:
            count = positions.get(pos, 0)
            pct = count / num_techs * 100 if num_techs > 0 else 0
            print(f"    • {pos}: {count} ({pct:.1f}%)")

def main():
    # Load data
    embeddings, df = load_data()
    
    # Perform clustering
    kmeans, clusters, distances_to_cluster, distances_to_nearest_other = perform_clustering_with_distances(embeddings)
    
    # Save cluster assignments
    cluster_df = save_cluster_assignments(df, clusters, distances_to_cluster, distances_to_nearest_other, kmeans)
    
    # Save boundary analysis
    boundaries_df = save_cluster_boundaries(cluster_df, kmeans, embeddings)
    
    # Save categorized techniques
    save_boundary_categorized(cluster_df)
    
    # Create visualization data
    create_boundary_visualization(cluster_df, embeddings)
    
    # Print summary
    print_summary(cluster_df, boundaries_df)
    
    print("\n" + "="*80)
    print("✓ All cluster files saved!")
    print("="*80)

if __name__ == '__main__':
    main()
