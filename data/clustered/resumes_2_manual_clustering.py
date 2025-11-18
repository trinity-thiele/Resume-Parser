import pandas as pd
# Manual clustering of 36 IT categories into logical groups

# Using raw data
input_filepath = 'data/raw/resumes_2.csv'
output_filepath = 'data/clustered/clustered_resumes_2.csv'

category_to_cluster = {
    # Cluster 0: Backend Development
    'Java Developer': 0,
    'Python Developer': 0,
    'DotNet Developer': 0,
    
    # Cluster 1: Frontend Development
    'Web Designing': 1,
    
    # Cluster 2: Full Stack Development
    # None
    
    # Cluster 3: Data Science & AI/ML
    'Data Science': 3,
    'Hadoop': 3,
    'ETL Developer': 3,
    'Business Analyst': 3,
    
    # Cluster 4: Database Administration
    'Database': 4,
    
    # Cluster 5: DevOps & Cloud
    'DevOps Engineer': 5,

    # Cluster 6: Quality Assurance
    'Automation Testing': 6,
    'Testing': 6,
    
    # Cluster 7: Security
    'Network Security Engineer': 7,
    
    # Cluster 8: Systems Administration
    # None
    
    # Cluster 9: Mobile Development
    # None
    
    # Cluster 10: Specialized Technologies
    'Blockchain': 10,
    'SAP Developer': 10,
    
    # Cluster 11: Engineering Leadership
    # None
    
    # Cluster 12: Content & Documentation
    # None
}

# Create reverse mapping (cluster to categories)
cluster_names = {
    0: 'Backend Development',
    1: 'Frontend Development',
    3: 'Data Science & AI/ML',
    4: 'Database Administration',
    5: 'DevOps & Cloud',
    6: 'Quality Assurance',
    7: 'Security',
    10: 'Specialized Technologies',
}

# Apply clustering to dataframe
df = pd.read_csv(input_filepath)

# Map categories to clusters
df['cluster'] = df['Category'].map(category_to_cluster)
df['cluster_name'] = df['cluster'].map(cluster_names)

# Show cluster distribution
print("="*60)
print("CLUSTER DISTRIBUTION")
print("="*60)
for cluster_id in sorted(cluster_names.keys()):
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}: {cluster_names[cluster_id]}")
    print(f"  Total resumes: {len(cluster_data)}")
    print(f"  Categories:")
    for cat, count in cluster_data['Category'].value_counts().items():
        print(f"    - {cat}: {count}")

# Save
df.to_csv(output_filepath, index=False)
print(f"Saved to {output_filepath}")