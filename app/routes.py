from flask import Blueprint, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import json
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
            
            # Extract PCA data for interactive visualization
            APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(APP_FOLDER, 'outputs', 'clustered_visitors_final.csv')
            
            if os.path.exists(output_path):
                df = pd.read_csv(output_path)
                pca_data = {
                    'data': df[['pca1', 'pca2', 'cluster']].to_dict('records'),
                    'cluster': df['cluster'].unique().tolist()
                }
                
                # Extract cluster sizes data
                cluster_counts = df['cluster'].value_counts().sort_index()
                cluster_sizes = {
                    'clusters': cluster_counts.index.tolist(),
                    'sizes': cluster_counts.values.tolist()
                }
                
                # Extract elbow data from the summary
                # Extract elbow data from the summary
                elbow_data = {
                    'k_values': list(range(1, 11)),
                    'inertias': [float(x) for x in summary.get('inertias', [...])],
                    'optimal_k': int(summary.get('optimal_k', 2))
                }

                
                return render_template('index.html', 
                    plots=True, 
                    summary=summary or [],
                    pca_data=pca_data,
                    cluster_sizes=cluster_sizes,
                    elbow_data=elbow_data,
                    pca_plot=summary['pca_plot'],
                    cluster_sizes_plot=summary['cluster_sizes_plot'],
                    elbow_plot=summary['elbow_plot']
                )
    
    return render_template('index.html', plots=False)

@main.route('/download', methods=['GET'])
def download_results():
    APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(APP_FOLDER, 'outputs', 'clustered_visitors_final.csv')
    
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
    output_path = os.path.join(APP_FOLDER, 'outputs', 'clustered_visitors_final.csv')
    
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        
        # Get cluster summary
        cluster_summary = []
        for cluster in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster]
            summary = {
                'cluster': int(cluster),
                'size': len(cluster_df),
                'percentage': round(len(cluster_df) / len(df) * 100, 2)
            }
            
            # Add most common values for each feature
            for col in df.columns:
                if col not in ['Visitor id', 'pca1', 'pca2', 'cluster']:
                    most_common = cluster_df[col].value_counts().index[0] if not cluster_df[col].empty else 'Unknown'
                    summary[f'{col}_most_common'] = most_common
            
            cluster_summary.append(summary)
        
        return jsonify({
            'clusters': cluster_summary,
            'total_visitors': len(df['Visitor id'].unique()),
            'features': [col for col in df.columns if col not in ['Visitor id', 'pca1', 'pca2', 'cluster']]
        })
    
    return jsonify({'error': 'No data available'}), 404
