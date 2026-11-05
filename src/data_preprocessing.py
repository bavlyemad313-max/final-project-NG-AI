"""
Data Preprocessing & Feature Engineering Pipeline
Project 02: Customer Lifetime Value (CLV) & Customer Segmentation Engine
"""
import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_CSV = os.path.join(DATA_DIR, 'customer_rfm_clv.csv')
CLEAN_TRANSACTIONS_CSV = os.path.join(DATA_DIR, 'clean_transactions.csv')

def load_raw_data():
    """
    Finds and loads the Online Retail II dataset from the raw directory.
    Supports .xlsx, .csv, or extracts sheets.
    """
    xlsx_files = glob.glob(os.path.join(RAW_DIR, '*.xlsx'))
    csv_files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
    
    if csv_files:
        print(f"Loading CSV data from {csv_files[0]}...")
        df = pd.read_csv(csv_files[0], encoding='ISO-8859-1')
        return df
    elif xlsx_files:
        excel_path = xlsx_files[0]
        print(f"Loading Excel file: {excel_path} (reading all sheets)...")
        # Online Retail II has two sheets: 'Year 2009-2010' and 'Year 2010-2011'
        xls = pd.ExcelFile(excel_path)
        dfs = []
        for sheet_name in xls.sheet_names:
            print(f"  Reading sheet: {sheet_name}...")
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            dfs.append(df_sheet)
        df = pd.concat(dfs, ignore_index=True)
        return df
    else:
        raise FileNotFoundError(f"No .xlsx or .csv found in {RAW_DIR}")

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw retail transactions:
    1. Standardizes column names.
    2. Drops rows with missing Customer ID.
    3. Excludes cancelled orders (Invoice starting with 'C').
    4. Filters positive Quantity and Unit Price.
    5. Calculates TotalSpend = Quantity * Price.
    6. Formats InvoiceDate as datetime.
    """
    print("Standardizing column names...")
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    
    # Handle possible column variations: customer_id or 'customer_id'
    col_mapping = {
        'customer_id': 'customer_id',
        'customerid': 'customer_id',
        'invoice': 'invoice',
        'invoiceno': 'invoice',
        'stockcode': 'stock_code',
        'description': 'description',
        'quantity': 'quantity',
        'invoicedate': 'invoice_date',
        'price': 'unit_price',
        'unitprice': 'unit_price',
        'country': 'country'
    }
    df = df.rename(columns=lambda c: col_mapping.get(c, c))
    
    initial_rows = len(df)
    print(f"Initial raw transactions count: {initial_rows:,}")
    
    # 1. Drop missing Customer ID
    df = df.dropna(subset=['customer_id']).copy()
    df['customer_id'] = df['customer_id'].astype(int).astype(str)
    print(f"Rows after dropping missing customer_id: {len(df):,}")
    
    # 2. Convert Invoice to string and filter cancellations
    df['invoice'] = df['invoice'].astype(str)
    df = df[~df['invoice'].str.startswith('C')].copy()
    print(f"Rows after removing cancellations: {len(df):,}")
    
    # 3. Filter positive Quantity and Unit Price
    df = df[(df['quantity'] > 0) & (df['unit_price'] > 0)].copy()
    print(f"Rows after filtering quantity > 0 and price > 0: {len(df):,}")
    
    # 4. Convert date
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    
    # 5. Compute total spend per line item
    df['total_spend'] = df['quantity'] * df['unit_price']
    
    return df

def build_rfm_and_clv(df: pd.DataFrame, prediction_days: int = 90) -> pd.DataFrame:
    """
    Builds RFM features and target variable with ZERO DATA LEAKAGE:
    - Splits transactions into Observation Period and Prediction Window.
    - Observation Period (< cut_off_date): Used to calculate RFM features.
    - Prediction Window (>= cut_off_date): Used to calculate target (spend in next 90 days).
    """
    max_date = df['invoice_date'].max()
    cut_off_date = max_date - timedelta(days=prediction_days)
    print(f"Dataset date range: {df['invoice_date'].min().strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"Time Cut-off Date: {cut_off_date.strftime('%Y-%m-%d')}")
    print(f"Observation window: Up to {cut_off_date.strftime('%Y-%m-%d')}")
    print(f"Prediction window: Next {prediction_days} days ({cut_off_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})")
    
    # Split datasets
    obs_df = df[df['invoice_date'] < cut_off_date].copy()
    pred_df = df[df['invoice_date'] >= cut_off_date].copy()
    
    print(f"Transactions in observation period: {len(obs_df):,}")
    print(f"Transactions in prediction period: {len(pred_df):,}")
    
    # Calculate RFM on observation period ONLY
    print("Computing RFM features on observation period...")
    rfm = obs_df.groupby('customer_id').agg(
        recency=('invoice_date', lambda dates: (cut_off_date - dates.max()).days),
        frequency=('invoice', 'nunique'),
        monetary=('total_spend', 'sum'),
        total_items=('quantity', 'sum'),
        unique_products=('stock_code', 'nunique'),
        first_purchase=('invoice_date', 'min'),
        last_purchase=('invoice_date', 'max'),
        country=('country', lambda x: x.mode()[0] if not x.empty else 'Unknown')
    ).reset_index()
    
    # Compute lifespan / tenure in days
    rfm['tenure_days'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days
    rfm['aov'] = rfm['monetary'] / rfm['frequency'] # Average Order Value
    rfm.drop(columns=['first_purchase', 'last_purchase'], inplace=True)
    
    # Calculate Target: actual spend in prediction window
    print("Computing target variable (spend in next 90 days)...")
    pred_spend = pred_df.groupby('customer_id')['total_spend'].sum().reset_index()
    pred_spend.rename(columns={'total_spend': 'target_spend_90d'}, inplace=True)
    
    # Merge target with RFM features
    customer_data = pd.merge(rfm, pred_spend, on='customer_id', how='left')
    customer_data['target_spend_90d'] = customer_data['target_spend_90d'].fillna(0.0)
    
    # Add binary churn / activity indicator for 90 days
    customer_data['purchased_next_90d'] = (customer_data['target_spend_90d'] > 0).astype(int)
    
    print(f"Total unique customers modeled: {len(customer_data):,}")
    print(f"Customers who purchased in next 90 days: {customer_data['purchased_next_90d'].sum():,} ({customer_data['purchased_next_90d'].mean()*100:.1f}%)")
    
    return customer_data

def run_pipeline():
    df_raw = load_raw_data()
    df_clean = clean_transactions(df_raw)
    
    # Save a clean transaction sample for rapid app preview
    os.makedirs(DATA_DIR, exist_ok=True)
    df_clean.head(1000).to_csv(CLEAN_TRANSACTIONS_CSV, index=False)
    
    customer_data = build_rfm_and_clv(df_clean, prediction_days=90)
    customer_data.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Preprocessed customer features saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    run_pipeline()
