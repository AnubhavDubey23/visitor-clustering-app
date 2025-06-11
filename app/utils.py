import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import requests
from sklearn.metrics import silhouette_score
import os

def find_optimal_k(data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
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

def select_diverse_features(df, selected_features, min_unique=2):
    return [col for col in selected_features if df[col].nunique() >= min_unique]

def safe_json_parse(value):
    try:
        return json.loads(value.replace("'", '"'))
    except Exception:
        return {}    

def notify_low_recency_users(rfm_df):
    webhook_url = "https://hook.eu2.make.com/3jdqmukqu5096k1y1ghh0vhbn7iso3fp"  # Replace with your real URL

    for _, row in rfm_df.iterrows():
        recency = row.get("Recency", 999)
        email = row.get("email") or row.get("Email")  # Adjust based on your column name

        if pd.notnull(email) and recency < 20:
            payload = {
                "email": email,
                "recency": recency
            }
            try:
                response = requests.post(webhook_url, json=payload)
                response.raise_for_status()
                print(f"✅ Notification sent to {email}")
            except Exception as e:
                print(f"❌ Failed to notify {email}: {str(e)}")

def process_rfm(df_rfm):
    df_rfm['First visit date'] = pd.to_datetime(df_rfm['First visit date'], dayfirst=True)
    df_rfm['Latest visit date'] = pd.to_datetime(df_rfm['Latest visit date'], dayfirst=True)

    rfm_df = df_rfm.groupby('Visitor id').agg({
        'Latest visit date': lambda x: (df_rfm['Latest visit date'].max() - x.max()).days,
        'Visitor id': 'count',
        'Email': 'first'
    }).rename(columns={
        'Latest visit date': 'Recency',
        'Visitor id': 'Frequency',
        'Email': 'Email'
    })
    rfm_df['Monetary'] = rfm_df['Frequency']

    # Debug print raw values before binning
    print("\nRecency values before binning:")
    print(rfm_df['Recency'].describe())
    
    print("\nFrequency values before binning:")
    print(rfm_df['Frequency'].describe())

    rfm_df['R'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm_df['F'] = pd.qcut(rfm_df['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm_df['M'] = pd.qcut(rfm_df['Monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Debug print after binning
    print("\nR scores distribution:")
    print(rfm_df['R'].value_counts().sort_index())
    
    print("\nF scores distribution:")
    print(rfm_df['F'].value_counts().sort_index())
    
    print("\nM scores distribution:")
    print(rfm_df['M'].value_counts().sort_index())

    rfm_df['RFM_Score'] = rfm_df['R'].astype(str) + rfm_df['F'].astype(str) + rfm_df['M'].astype(str)

    print("\nRFM Score distribution:")
    print(rfm_df['RFM_Score'].value_counts().head(20))

    def segment(rfm):
        if rfm['R'] >= 4 and rfm['F'] >= 4:
            return 'Champions'
        elif rfm['R'] >= 3 and rfm['F'] >= 3:
            return 'Loyal Customers'
        elif rfm['R'] >= 4:
            return 'Recent Customers'
        elif rfm['F'] >= 4:
            return 'Frequent Visitors'
        else:
            return 'Others'
    
    rfm_df['Segment'] = rfm_df.apply(segment, axis=1)
    return rfm_df

def process_file(input_path):
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    df['device_data'] = df['meta data (device data)'].apply(safe_json_parse)
    df['network_data'] = df['meta data (network data)'].apply(safe_json_parse)

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

    APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(APP_FOLDER, '..', 'static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    def build_features(df):
        if 'Visitor id' in df.columns and df['Visitor id'].dropna().astype(str).str.strip().ne('').any():
            df['user_key'] = 'visitor_' + df['Visitor id'].astype(str)
            visit_counts = df['user_key'].value_counts().to_dict()
            df['visit_count'] = df['user_key'].map(visit_counts)
            df['visitor_type'] = df['visit_count'].apply(lambda x: 'Returning' if x > 1 else 'New')

            rfm_df=process_rfm(df)

            # send_email_to_low_recency_users(rfm_df)
            # After RFM calculation is done and RFM dataframe is ready
            notify_low_recency_users(rfm_df)


            features = ['network_org', 'State', 'device_browser_details', 'device_os', 'visitor_type', 'Product name']

            df = df.merge(rfm_df[['Recency', 'Frequency', 'Monetary', 'R', 'F', 'M', 'RFM_Score', 'Segment']],
                              left_on='Visitor id', right_index=True, how='left')

            features += ['RFM_Score']
            # ===== ADD RFM SCORE DISTRIBUTION PLOT =====
            plt.figure(figsize=(12,6))
            rfm_df['RFM_Score'].value_counts().sort_index().plot(kind='bar', color='skyblue')
            plt.title('RFM Score Distribution')
            plt.xlabel('RFM Score')
            plt.ylabel('Number of Customers')
            rfm_plot_path = os.path.join(plot_dir, 'rfm_score_distribution.png')
            plt.savefig(rfm_plot_path)
            plt.close()
        else:
            def get_user_key(row):
                if 'Registered User Id' in df.columns and pd.notnull(row.get('Registered User Id')) and str(row['Registered User Id']).strip() != '':
                    return f"reg_{row['Registered User Id']}"
                elif 'Unregistered User Id' in df.columns and pd.notnull(row.get('Unregistered User Id')) and str(row['Unregistered User Id']).strip() != '':
                    return f"unreg_{row['Unregistered User Id']}"
                else:
                    return 'unknown'
            df['user_key'] = df.apply(get_user_key, axis=1)
            user_counts = df['user_key'].value_counts().to_dict()
            df['user_count'] = df['user_key'].map(user_counts)

            features = ['city', 'network_org', 'device_browser_details', 'device_os', 'user_count']

        return df, features


    df['device_os'] = df['device_os'].str.strip('"')

    # candidate_cols = ['network_org', 'State', 'device_browser_details', 'device_os', 'visitor_type','Product name']

    df,features=build_features(df)
    # features = select_strong_features(df, features_2, top_n=5, threshold=0.5)
    clustering_df = df[features].copy()

    for col in clustering_df.select_dtypes(include='object').columns:
        top_5 = clustering_df[col].value_counts().nlargest(5).index
        clustering_df[col] = clustering_df[col].apply(lambda x: x if x in top_5 else 'Other')

    df_encoded = pd.get_dummies(clustering_df, drop_first=True).fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)


    # ==== KMeans ====
    optimal_k, inertias = find_optimal_k(scaled_data)


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

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(scaled_data)

    # Compute silhouette score for KMeans
    silhouette_kmeans = silhouette_score(scaled_data, kmeans_labels)


    # ==== HDBSCAN ====
    hdb = hdbscan.HDBSCAN(min_cluster_size=10)
    hdb_labels = hdb.fit_predict(scaled_data)

    # Compute silhouette score for HDBSCAN (only on non-noise points)
    mask = hdb_labels != -1
    if len(set(hdb_labels[mask])) > 1:# need at least 2 samples to compute score
        silhouette_hdbscan = silhouette_score(scaled_data[mask], hdb_labels[mask])
    else:
        silhouette_hdbscan = None


    # PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    clustering_df['pca1'] = pca_data[:, 0]
    clustering_df['pca2'] = pca_data[:, 1]
    clustering_df['kmeans_cluster'] = kmeans_labels
    clustering_df['hdbscan_cluster'] = hdb_labels

    # ==== Cluster Size Bar Plot ====
    def plot_cluster_sizes(cluster_col, title, filename):
        valid_df = clustering_df[clustering_df[cluster_col] != -1]
        cluster_counts = valid_df[cluster_col].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cluster_counts.index, cluster_counts.values,
                       color=sns.color_palette("husl", len(cluster_counts)))
        plt.title(title)
        plt.xlabel("Cluster")
        plt.ylabel("Number of Visitors")
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 5, f'{int(height)}', ha='center')
        plt.tight_layout()
        out_path = os.path.join(plot_dir, filename)
        plt.savefig(out_path)
        plt.close()

    plot_cluster_sizes('kmeans_cluster', "KMeans Cluster Distribution", "cluster_sizes_kmeans.png")
    plot_cluster_sizes('hdbscan_cluster', "HDBSCAN Cluster Distribution", "cluster_sizes_hdbscan.png")

    # ==== PCA Plot ====
    def plot_pca(cluster_col, title, filename):
        valid_df = clustering_df[clustering_df[cluster_col] != -1]
        plt.figure(figsize=(10, 8))
        for cluster in sorted(valid_df[cluster_col].unique()):
            subset = valid_df[valid_df[cluster_col] == cluster]
            plt.scatter(subset['pca1'], subset['pca2'], label=f'Cluster {cluster}', s=60, alpha=0.7)
        plt.title(title)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(plot_dir, filename)
        plt.savefig(out_path)
        plt.close()

    plot_pca('hdbscan_cluster', 'HDBSCAN PCA Plot', 'pca_plot_hdbscan.png')

    # ==== KMeans PCA Plot (Enhanced Style) ====
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        clustering_df['pca1'],
        clustering_df['pca2'],
        c=clustering_df['kmeans_cluster'],
        cmap='tab10',
        s=50,
        alpha=0.8,
        edgecolors='w',
        linewidth=0.5
    )
    plt.title('PCA of KMeans Visitor Clusters', fontsize=14)
    plt.xlabel('PCA 1', fontsize=12)
    plt.ylabel('PCA 2', fontsize=12)
    cbar = plt.colorbar(scatter, ticks=range(optimal_k))
    cbar.set_label('Cluster', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    kmeans_pca_path = os.path.join(plot_dir, 'pca_plot_kmeans.png')
    plt.savefig(kmeans_pca_path, dpi=100)
    plt.close()


    # ==== Cluster Summary ====
    RENAME_MAP = {
        'State': 'State',
        'network_org': 'Organization of Network',
        'visitor_type': 'Visitor Type',
        'device_browser_details': 'Browser Used',
        'device_os': 'Type of Device',
        'Product name': 'Name of Product',
        'city':'City',
        'age':'Year of Birth',

    }

    selected_renames = {col: RENAME_MAP[col] for col in features if col in RENAME_MAP}

    kmeans_summary = clustering_df.groupby('kmeans_cluster').agg({
        col: lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan for col in features
    }).rename(columns=selected_renames).reset_index()

    hdbscan_summary = clustering_df[clustering_df['hdbscan_cluster'] != -1].groupby('hdbscan_cluster').agg({
        col: lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan for col in features
    }).rename(columns=selected_renames).reset_index()

    # ==== Output CSV ====
    output_folder = os.path.join(APP_FOLDER, 'outputs')
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'clustered_visitors_combined.csv')
    clustering_df.to_csv(output_path, index=False)

    return {
        'summary_kmeans': kmeans_summary.to_dict(orient='records'),
        'summary_hdbscan': hdbscan_summary.to_dict(orient='records'),
        'csv_path': output_path,
        'elbow_plot': 'plots/elbow_plot.png',
        'pca_plot_kmeans': 'plots/pca_plot_kmeans.png',
        'pca_plot_hdbscan': 'plots/pca_plot_hdbscan.png',
        'cluster_sizes_kmeans': 'plots/cluster_sizes_kmeans.png',
        'cluster_sizes_hdbscan': 'plots/cluster_sizes_hdbscan.png',
        'rfm_score_distribution': 'plots/rfm_score_distribution.png', 
        'silhouette_kmeans': silhouette_kmeans,
        'silhouette_hdbscan': silhouette_hdbscan,
        'optimal_k': optimal_k,
        'inertias': inertias,
        'features': features,
        'num_features': len(features)
    }
