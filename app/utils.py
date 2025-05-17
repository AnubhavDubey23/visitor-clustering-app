import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')  # For server use, no GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def find_optimal_k(data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    x = np.arange(1, max_k + 1)
    y = np.array(inertias)
    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    line = line / np.linalg.norm(line)

    distances = []
    for i in range(len(x)):
        point = np.array([x[i] - x[0], y[i] - y[0]])
        proj_len = np.dot(point, line)
        proj_point = np.array([x[0], y[0]]) + proj_len * line
        distance = np.linalg.norm(np.array([x[i], y[i]]) - proj_point)
        distances.append(distance)

    optimal_k = x[np.argmax(distances)]
    return optimal_k, inertias

def select_strong_features(df, candidate_cols, top_n, threshold):
    selected_features = []
    total_rows = len(df)

    for col in candidate_cols:
        top_counts = df[col].value_counts().nlargest(top_n).sum()
        top_percent = top_counts / total_rows
        if top_percent >= threshold:
            selected_features.append(col)

    return selected_features

def select_diverse_features(df, candidate_cols, top_n, threshold):
    selected_features = []
    total_rows = len(df)

    for col in candidate_cols:
        top_counts = df[col].value_counts().nlargest(top_n).sum()
        top_percent = top_counts / total_rows
        if top_percent < threshold:
            selected_features.append(col)

    return selected_features

def process_file(input_path):
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    def safe_json_parse(val):
        if isinstance(val, dict):
            return val
        if pd.isnull(val):
            return {}
        try:
            return json.loads(val.replace("'", '"'))
        except json.JSONDecodeError:
            return {}

    df['device_data'] = df['Meta data (device data)'].apply(safe_json_parse)
    df['network_data'] = df['Meta data (network data)'].apply(safe_json_parse)

    df['device_browser_details'] = df['device_data'].apply(lambda x: x.get('browser_details', np.nan))
    df['device_os'] = df['device_data'].apply(lambda x: x.get('os', np.nan))
    df['network_org'] = df['network_data'].apply(lambda x: x.get('org', np.nan))
    df['network_country'] = df['network_data'].apply(lambda x: x.get('country', np.nan))

    df['device_os'] = df['device_os'].fillna('unknown').replace('', 'unknown')

    known_browsers = ["Google Chrome", "Samsung Internet", "Android WebView", "Brave", "Mozilla"]
    def extract_browser_name(details):
        if pd.isnull(details) or details.strip() == "":
            return "unknown"
        for browser in known_browsers:
            if browser.lower() in details.lower():
                return browser
        return "unknown"

    df['device_browser_details'] = df['device_browser_details'].apply(extract_browser_name)

    visit_counts = df.groupby('Visitor id').size().reset_index(name='num_visits')
    df = df.merge(visit_counts, on='Visitor id')
    df['visitor_type'] = df['num_visits'].apply(lambda x: 'Returning' if x > 1 else 'New')

    df['device_os'] = df['device_os'].str.strip('"')

    candidate_cols = ['network_org', 'State', 'device_browser_details', 'device_os', 'visitor_type','Product name']
    strong_features = select_strong_features(df, candidate_cols, top_n=5, threshold=0.5)
    diverse_features = select_diverse_features(df, candidate_cols, top_n=3, threshold=0.7)
    # features = [col for col in strong_features if col in diverse_features]
    features=strong_features

    clustering_df = df[['Visitor id'] + features].copy()

    for col in features:
        top_5 = clustering_df[col].value_counts().nlargest(5).index
        clustering_df[col] = clustering_df[col].apply(lambda x: x if x in top_5 else 'Other')

    df_encoded = pd.get_dummies(clustering_df[features], drop_first=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)

    optimal_k, inertias = find_optimal_k(scaled_data)

    APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(APP_FOLDER, '..', 'static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Create plots with improved styling
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, marker='o', linestyle='-', linewidth=2, color='#3B82F6')
    plt.title("Elbow Method For Optimal k", fontsize=16)
    plt.xlabel("Number of clusters", fontsize=12)
    plt.ylabel("SSE", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Highlight the optimal k
    plt.scatter(optimal_k, inertias[optimal_k-1], s=200, c='#EF4444', zorder=5, edgecolors='white', linewidth=2)
    plt.annotate(f'Optimal k = {optimal_k}', 
                 xy=(optimal_k, inertias[optimal_k-1]),
                 xytext=(optimal_k + 1, inertias[optimal_k-1] + 5000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12)
    
    elbow_path = os.path.join(plot_dir, 'elbow_plot.png')
    plt.savefig(elbow_path, dpi=100, bbox_inches='tight')
    plt.close()

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    clustering_df['pca1'] = pca_data[:, 0]
    clustering_df['pca2'] = pca_data[:, 1]
    clustering_df['cluster'] = labels

    # Cluster sizes plot with improved styling
    plt.figure(figsize=(10, 6))
    cluster_counts = clustering_df['cluster'].value_counts().sort_index()
    
    # Use a colorful palette
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', 
              '#FF9F40', '#8AC926', '#1982C4', '#6A4C93', '#F15BB5']
    
    bars = plt.bar(cluster_counts.index, cluster_counts.values, 
                   color=[colors[i % len(colors)] for i in cluster_counts.index])
    
    plt.title("Visitor Cluster Distribution", fontsize=16)
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel("Number of Visitors", fontsize=12)
    plt.xticks(cluster_counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    cluster_size_path = os.path.join(plot_dir, 'cluster_sizes.png')
    plt.savefig(cluster_size_path, dpi=100, bbox_inches='tight')
    plt.close()

    # PCA plot with improved styling
    plt.figure(figsize=(10, 8))
    
    # Create a scatter plot for each cluster with a different color
    for i in range(optimal_k):
        cluster_data = clustering_df[clustering_df['cluster'] == i]
        plt.scatter(cluster_data['pca1'], cluster_data['pca2'], 
                    c=[colors[i % len(colors)]], 
                    label=f'Cluster {i}',
                    s=70, alpha=0.8, edgecolors='w', linewidth=0.5)
    
    plt.title('PCA of Visitor Clusters', fontsize=16)
    plt.xlabel('PCA 1', fontsize=12)
    plt.ylabel('PCA 2', fontsize=12)
    plt.legend(title='Clusters', title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add a light background grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Improve the layout
    plt.tight_layout()
    
    pca_path = os.path.join(plot_dir, 'pca_plot.png')
    plt.savefig(pca_path, dpi=100, bbox_inches='tight')
    plt.close()

    RENAME_MAP = {
        'State': 'State',
        'network_org': 'Organization of Network',
        'visitor_type': 'Visitor Type',
        'device_browser_details': 'Browser Used',
        'device_os': 'Type of Device',
        'Product name':'Name of Product'
    }

    selected_renames = {col: RENAME_MAP[col] for col in features if col in RENAME_MAP}

    summary = clustering_df.groupby('cluster').agg({
        col: lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan for col in features
    }).rename(columns=selected_renames).reset_index()

    output_folder = os.path.join(APP_FOLDER, 'outputs')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'clustered_visitors_final.csv')
    clustering_df.to_csv(output_path, index=False)

    return {
        'summary': summary.to_dict(orient='records'),
        'csv_path': output_path,
        'elbow_plot': 'plots/elbow_plot.png',
        'cluster_sizes_plot': 'plots/cluster_sizes.png',
        'pca_plot': 'plots/pca_plot.png',
        'optimal_k': optimal_k,
        'inertias': inertias,
        'features': features,
        'num_features': len(features)
    }
