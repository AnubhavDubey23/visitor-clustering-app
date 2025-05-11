import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def process_file(input_path):
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # Your existing code here (copy-paste cleanly)
    # ✅ Extract device type
    def infer_device_type(meta_data):
        try:
            if pd.isna(meta_data):
                return 'unknown'
            parsed = json.loads(meta_data)
            os_info = parsed.get('os', '').lower()
            if 'android' in os_info or 'ios' in os_info:
                return 'mobile'
            elif 'windows' in os_info or 'mac' in os_info or 'linux' in os_info:
                return 'desktop'
            else:
                return 'unknown'
        except:
            return 'unknown'

    df['device_type'] = df['Meta data (device data)'].apply(infer_device_type)

    def extract_isp(x):
        try:
            if pd.isna(x):
                return 'unknown'
            return json.loads(x).get('org', 'unknown')
        except:
            return 'unknown'

    df['network_provider'] = df['Meta data (network data)'].apply(extract_isp)

    visit_counts = df.groupby('Visitor id').size().reset_index(name='num_visits')
    df = df.merge(visit_counts, on='Visitor id')
    df['visitor_type'] = df['num_visits'].apply(lambda x: 'Returning' if x > 1 else 'New')

    top5_network = df['network_provider'].value_counts().nlargest(5).index
    df['network_provider'] = df['network_provider'].where(df['network_provider'].isin(top5_network), 'Other')

    top2_brands = df['Brand name'].value_counts().nlargest(2).index
    df['brand_affinity'] = df['Brand name'].where(df['Brand name'].isin(top2_brands), 'Other')

    top2_devices = df['device_type'].value_counts().nlargest(2).index
    df['device_type'] = df['device_type'].where(df['device_type'].isin(top2_devices), 'unknown')

    top5_cities = df['City'].value_counts().nlargest(5).index
    df['City'] = df['City'].where(df['City'].isin(top5_cities), 'Other')

    df['State'] = df['State'].astype(str)
    top5_states = df['State'].value_counts().nlargest(5).index
    df['State'] = df['State'].where(df['State'].isin(top5_states), 'Other')

    visitor_df = df.groupby('Visitor id').agg({
        'device_type': 'first',
        'network_provider': 'first',
        'City': 'first',
        'State': 'first',
        'visitor_type': 'first',
        'brand_affinity': 'first'
    }).reset_index()

    visitor_df_encoded = pd.get_dummies(visitor_df, columns=[
        'device_type', 'network_provider', 'City', 'State',
        'visitor_type', 'brand_affinity'
    ])

    features = [col for col in visitor_df_encoded.columns if col != 'Visitor id']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(visitor_df_encoded[features])

    kmeans = KMeans(n_clusters=4, random_state=42)
    visitor_df['cluster'] = kmeans.fit_predict(scaled_features)

    APP_FOLDER = os.path.dirname(os.path.abspath(__file__))  # this gives ".../app"

# Set outputs folder inside app
    output_folder = os.path.join(APP_FOLDER, 'outputs')
    os.makedirs(output_folder, exist_ok=True)

    # output_path = 'clustered_visitors_final.csv'
    # visitor_df.to_csv(output_path, index=False)

    output_path = os.path.join(output_folder, 'clustered_visitors_final.csv')
    visitor_df.to_csv(output_path, index=False)  # Example save    

    return output_path
