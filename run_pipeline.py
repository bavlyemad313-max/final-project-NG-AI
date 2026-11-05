"""
End-to-End Orchestrator Pipeline (v2 - Two-Stage Architecture)
Executes: Preprocessing -> Unsupervised Segmentation -> Two-Stage CLV (Classification + Regression)
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import joblib
import pandas as pd
import numpy as np

from src.data_preprocessing import load_raw_data, clean_transactions, build_rfm_and_clv
from src.segmentation import train_segmentation
from src.clv_prediction import run_clv_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def run_full_pipeline():
    print("=" * 60)
    print("STEP 1: PREPROCESSING & RFM FEATURE ENGINEERING")
    print("=" * 60)
    customer_csv = os.path.join(DATA_DIR, 'customer_rfm_clv.csv')
    if not os.path.exists(customer_csv):
        df_raw = load_raw_data()
        df_clean = clean_transactions(df_raw)
        df_customer = build_rfm_and_clv(df_clean, prediction_days=90)
        df_customer.to_csv(customer_csv, index=False)
    else:
        print(f"Loaded existing {customer_csv}")
        df_customer = pd.read_csv(customer_csv)

    print("\n" + "=" * 60)
    print("STEP 2: UNSUPERVISED CLUSTERING & PCA")
    print("=" * 60)
    df_segmented, summary_table = train_segmentation(df_customer, n_clusters=4)

    print("\n" + "=" * 60)
    print("STEP 3: TWO-STAGE CLV (CLASSIFICATION + REGRESSION)")
    print("=" * 60)
    tuned_clf, tuned_reg, clf_results = run_clv_pipeline(df_segmented)

    # Merge segmentation columns into master
    master_path = os.path.join(DATA_DIR, 'customer_profiles_master.csv')
    df_final = pd.read_csv(master_path)

    # Add segmentation columns if not already present
    if 'segment_name' not in df_final.columns:
        df_final['cluster'] = df_segmented['cluster'].values
        df_final['segment_name'] = df_segmented['segment_name'].values
        df_final['pca_x'] = df_segmented['pca_x'].values
        df_final['pca_y'] = df_segmented['pca_y'].values
        df_final.to_csv(master_path, index=False)

    print("\n" + "=" * 60)
    print("SUCCESS: Complete end-to-end pipeline finished.")
    print(f"Master dataset saved to: {master_path}")
    print(f"Models and transformers saved in: {MODELS_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    run_full_pipeline()
