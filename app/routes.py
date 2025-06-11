from flask import Blueprint, render_template, request, send_file, jsonify
import pandas as pd
import os
from .utils import process_file

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploaded.csv')
            file.save(filepath)
            summary = process_file(filepath)

            APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(APP_FOLDER, 'outputs', 'clustered_visitors_combined.csv')

            # Initialize variables with default values
            pca_data = {'kmeans': [], 'hdbscan': []}
            cluster_sizes = {'kmeans': {'clusters': [], 'sizes': []}, 'hdbscan': {'clusters': [], 'sizes': []}}
            elbow_data = {'k_values': [], 'inertias': [], 'optimal_k': 0}
            show_rfm = False
            rfm_scores = []
            rfm_counts = []

            if os.path.exists(output_path):
                df = pd.read_csv(output_path)

                # PCA data
                pca_data = {
                    'kmeans': df[['pca1', 'pca2', 'kmeans_cluster']].rename(
                        columns={'pca1': 'x', 'pca2': 'y', 'kmeans_cluster': 'cluster'}
                    ).to_dict('records'),
                    'hdbscan': df[df['hdbscan_cluster'] != -1][['pca1', 'pca2', 'hdbscan_cluster']].rename(
                        columns={'pca1': 'x', 'pca2': 'y', 'hdbscan_cluster': 'cluster'}
                    ).to_dict('records'),
                }

                # Cluster sizes
                if 'kmeans_cluster' in df.columns:
                    kmeans_counts = df['kmeans_cluster'].value_counts().sort_index()
                    cluster_sizes['kmeans']['clusters'] = kmeans_counts.index.tolist()
                    cluster_sizes['kmeans']['sizes'] = kmeans_counts.values.tolist()

                if 'hdbscan_cluster' in df.columns:
                    hdbscan_counts = df[df['hdbscan_cluster'] != -1]['hdbscan_cluster'].value_counts().sort_index()
                    cluster_sizes['hdbscan']['clusters'] = hdbscan_counts.index.tolist()
                    cluster_sizes['hdbscan']['sizes'] = hdbscan_counts.values.tolist()

                elbow_data = {
                    'k_values': list(range(1, 11)),
                    'inertias': [float(x) for x in summary.get('inertias', [])],
                    'optimal_k': int(summary.get('optimal_k', 2))
                }

                # RFM data - only if column exists
                if 'RFM_Score' in df.columns:
                    rfm_score_counts = df['RFM_Score'].value_counts()
                    rfm_scores = rfm_score_counts.index.tolist()
                    rfm_counts = rfm_score_counts.values.tolist()
                    show_rfm = len(rfm_scores) > 0  # Only show if we have scores

            return render_template(
                'index.html',
                plots=True,
                show_rfm=show_rfm,
                summary_kmeans=summary.get('summary_kmeans', []),
                summary_hdbscan=summary.get('summary_hdbscan', []),
                pca_data=pca_data,
                cluster_sizes=cluster_sizes,
                elbow_data=elbow_data,
                pca_plot_kmeans=summary['pca_plot_kmeans'],
                pca_plot_hdbscan=summary['pca_plot_hdbscan'],
                cluster_sizes_kmeans=summary['cluster_sizes_kmeans'],
                cluster_sizes_hdbscan=summary['cluster_sizes_hdbscan'],
                elbow_plot=summary['elbow_plot'],
                rfm_score_distribution=summary.get('rfm_score_distribution'),
                rfm_scores=rfm_scores,
                rfm_counts=rfm_counts,
                silhouette_kmeans=summary.get('silhouette_kmeans'),
                silhouette_hdbscan=summary.get('silhouette_hdbscan')
            )

    return render_template('index.html', plots=False)


@main.route('/download', methods=['GET'])
def download_results():
    APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(APP_FOLDER, 'outputs', 'clustered_visitors_combined.csv')

    if os.path.exists(output_path):
        return send_file(output_path,
                         mimetype='text/csv',
                         download_name='clustered_visitors.csv',
                         as_attachment=True)
    else:
        return "No results file available. Please process data first.", 404

@main.route('/api/cluster-data', methods=['GET'])
def get_cluster_data():
    APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(APP_FOLDER, 'outputs', 'clustered_visitors_combined.csv')

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)

        result = {
            'total_visitors': len(df['Visitor id'].unique()) if 'Visitor id' in df.columns else len(df),
            'features': [col for col in df.columns if col not in ['Visitor id', 'pca1', 'pca2', 'kmeans_cluster', 'hdbscan_cluster']],
            'clusters': {
                'kmeans': [],
                'hdbscan': []
            }
        }

        for cluster in sorted(df['kmeans_cluster'].dropna().unique()):
            cluster_df = df[df['kmeans_cluster'] == cluster]
            summary = {
                'cluster': int(cluster),
                'size': len(cluster_df),
                'percentage': round(len(cluster_df) / len(df) * 100, 2)
            }
            for col in result['features']:
                most_common = cluster_df[col].value_counts().idxmax() if not cluster_df[col].empty else 'Unknown'
                summary[f'{col}_most_common'] = most_common
            result['clusters']['kmeans'].append(summary)

        if 'hdbscan_cluster' in df.columns:
            for cluster in sorted(df['hdbscan_cluster'].dropna().unique()):
                if cluster == -1:
                    continue
                cluster_df = df[df['hdbscan_cluster'] == cluster]
                summary = {
                    'cluster': int(cluster),
                    'size': len(cluster_df),
                    'percentage': round(len(cluster_df) / len(df) * 100, 2)
                }
                for col in result['features']:
                    most_common = cluster_df[col].value_counts().idxmax() if not cluster_df[col].empty else 'Unknown'
                    summary[f'{col}_most_common'] = most_common
                result['clusters']['hdbscan'].append(summary)

        return jsonify(result)

    return jsonify({'error': 'No data available'}), 404
