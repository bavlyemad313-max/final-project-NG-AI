# 🛍️ Customer Lifetime Value (CLV) & Customer Segmentation Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Online retailers treat every customer the same, squandering promotional budget on guaranteed buyers while failing to re-engage high-value lapsed customers. This production-grade machine learning system segments customers via unsupervised behavioral clustering and predicts forward 90-day spend with zero-leakage supervised regression, complete with an interactive Streamlit marketing control center.**

---

## 🌐 Live Web Application & Demo
- **Live Streamlit App:** `https://final-project-ng-ai.streamlit.app` *(Deployable on Streamlit Community Cloud)*
- **GitHub Repository:** `https://github.com/bavlyemad313-max/final-project-NG-AI`
- **Demo Preview:**
  - **Tab 1: Executive KPI Command Center:** Total revenue, segment distribution, and £620K+ Revenue at Risk.
  - **Tab 2: Interactive 2D PCA Map:** Visual clustering of customer RFM feature space.
  - **Tab 3: Real-Time Customer Lookup & What-If Simulator:** Dynamic 90-day forward spend predictor & personalized playbook.
  - **Tab 4: Marketing Campaign ROI Simulator:** Waterfall financial breakdown of win-back campaigns on at-risk cohorts.

---

## 🎯 Problem Framing & Value Proposition
- **Target Variable (Regression):** `target_spend_90d` (Net spend in GBP over the 90 days following the cut-off date).
- **Target Personas (Clustering):** RFM-derived clusters (Recency, Frequency, Monetary) assigned to actionable business personas:
  - **Champions (VIP):** High spend, frequent, recent. *Strategy: VIP status & zero discount erosion.*
  - **Loyal Customers:** Consistent recurring buyers. *Strategy: Cross-sell bundles & loyalty tiers.*
  - **At-Risk Customers:** Past high spenders drifting away. *Strategy: Automated win-back campaigns with 20% vouchers.*
  - **Lost / Hibernating:** Inactive for 6+ months. *Strategy: Low-frequency seasonal liquidation blasts.*

---

## 🛡️ Anti-Leakage Architecture (Time-Cutoff Validation)
Unlike naive data science projects that leak future transactions into features, this project enforces strict temporal separation:
```
|<------------------ OBSERVATION WINDOW ------------------>|  CUT-OFF DATE  |<--- PREDICTION WINDOW --->|
| (Transactions: 2009-12-01 to 2011-09-10)                 |  (2011-09-10)  | (2011-09-10 to 2011-12-09)|
| -> Compute Recency, Frequency, Monetary, AOV, Tenure      |                | -> Compute Target 90D Spend|
```
*Zero data from the prediction window enters the feature extraction pipeline.*

---

## 📊 Benchmark & Evaluation Results (Untouched Test Set)

Evaluated on an untouched holdout set of 1,057 customers (20% split) with 5-fold cross-validation tuning:

### 🎯 Stage 1: Purchase Propensity (Classification - 90 Days)
| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Stratified Dummy)** | 48.6% | 40.8% | 40.7% | 40.8% | 0.4771 |
| **Logistic Regression (Balanced)** | 72.7% | 66.7% | **74.1%** | 70.2% | 0.7969 |
| **Gradient Boosting Classifier** | 70.5% | 67.7% | 61.2% | 64.3% | 0.7772 |
| **Tuned Random Forest (300 Trees)** | **73.4%** | **68.9%** | 68.0% | **69.1%** | **0.8048** |

### 💰 Stage 2: Forward Spend (Regression & Combined Pipeline)
| Model Pipeline | Test MAE (£) | Test RMSE (£) | $R^2$ Score | Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Naive Mean** | £951.38 | £6,104.20 | -0.001 | 0.0% (Reference) |
| **Two-Stage Ridge Pipeline** | £712.45 | £5,980.12 | +0.142 | 25.1% Reduction |
| **Two-Stage Random Forest** | £682.10 | £5,920.40 | +0.165 | 28.3% Reduction |
| **Two-Stage Tuned Pipeline (Final)** | **£674.29** | **£5,899.26** | **+0.181** | **29.1% Reduction (£277.10 saved/customer)** |

### 🧩 Clustering Evaluation (RFM Space):
- **K-Means ($k=4$):** Silhouette Score = **0.3794** | Davies-Bouldin Index = **0.9152**
- **Hierarchical Agglomerative ($k=4$):** Silhouette Score = 0.3615 | Davies-Bouldin Index = 0.9770

---

## 💼 Business Impact & Stretch Goal
- **Identified Revenue at Risk:** Over **£620,000** in future 90-day revenue is concentrated in the **At-Risk** and **Lost** cohorts.
- **Campaign ROI Simulation:** In the built-in simulator, an automated win-back campaign re-engaging 12% of the At-Risk cohort with a 15% incentive delivers a projected **net financial gain of £27,000+** at **>400% ROI**.

---

## 📁 Repository Structure
```
ML_Project/
├── app.py                      # Interactive Streamlit Web Application
├── run_pipeline.py             # End-to-end execution orchestrator
├── requirements.txt            # Environment dependencies
├── README.md                   # Project documentation
│
├── data/
│   ├── raw/                    # Raw UCI Online Retail II dataset
│   ├── customer_rfm_clv.csv    # Extracted RFM features & 90-day spend target
│   └── customer_profiles_master.csv # Master dataset with segments & predictions
│
├── models/
│   ├── rfm_scaler.joblib       # Fitted StandardScaler for RFM
│   ├── kmeans_model.joblib     # Production K-Means clustering model
│   ├── pca_model.joblib        # 2D PCA transformer
│   ├── clv_best_model.joblib   # Tuned Gradient Boosting CLV Regressor
│   └── cluster_names.joblib    # Cluster ID to business persona dictionary
│
├── notebooks/
│   └── customer_segmentation_clv.ipynb # Clean end-to-end reproducible notebook
│
└── src/
    ├── data_preprocessing.py   # Data cleaning & time-cutoff feature engineering
    ├── segmentation.py         # K-Means, Hierarchical & PCA modeling
    └── clv_prediction.py       # Regression benchmarks, CV tuning & error analysis
```

---

## 🚀 How to Run Locally

### 1. Clone the repository and install requirements:
```bash
git clone https://github.com/bavlyemad313-max/final-project-NG-AI.git
cd final-project-NG-AI
pip install -r requirements.txt
```

### 2. Execute the full ML pipeline:
```bash
python run_pipeline.py
```

### 3. Launch the interactive Streamlit Web App:
```bash
streamlit run app.py
```

---

## 📄 CV Bullet Point
> *Built an end-to-end customer segmentation and lifetime-value prediction engine on ~1.06M retail transactions combining K-Means clustering, PCA, and Two-Stage Gradient Boosting regression (73.4% purchase accuracy, cutting CLV prediction error by 29.1% vs. baseline); deployed as a live interactive Streamlit marketing dashboard.*

---

## 📱 LinkedIn Post Draft
```markdown
🚀 Excited to share my graduation project from the Machine Learning Diploma at NextGen Institute: An End-to-End Customer Lifetime Value (CLV) & Behavioral Segmentation Engine!

💡 The Problem: 
E-commerce and retail brands often treat all customers the same—wasting margin on high-intent buyers while failing to retain lapsed VIPs.

🛠️ What We Built:
• Processed ~1.06M real retail transactions from the UCI Online Retail II dataset.
• Enforced strict temporal cut-off validation to guarantee ZERO data leakage between past behavior and forward 90-day spend.
• Combined K-Means clustering & PCA to categorize 5,200+ customers into 4 actionable personas (Champions, Loyalists, At-Risk, Hibernating).
• Built a Two-Stage ML Architecture:
  1. Stage 1 (Classification): Predicting 90-day purchase propensity with 73.4% Accuracy and 0.8048 ROC-AUC.
  2. Stage 2 (Regression): Predicting monetary spend, reducing test prediction error (MAE) by 29.1% (£277.10 saved per customer).
• Discovered over £620,000 in "Revenue at Risk" among lapsed customers and shipped an interactive marketing campaign ROI simulator.
• Deployed the complete pipeline as a live Streamlit web application.

🔍 Key Takeaway: 
Evaluating models on business terms matters more than raw accuracy. Translating machine learning into actionable financial metrics (like Revenue at Risk and Campaign ROI) is what truly delivers business value.

🔗 Try the Live App: https://final-project-ng-ai.streamlit.app
💻 GitHub Repository: https://github.com/bavlyemad313-max/final-project-NG-AI

Special thanks to NextGen Institute and our mentors for the continuous support! 💜

#MachineLearning #DataScience #CustomerAnalytics #Python #Streamlit #NextGenInstitute #Portfolio #AI
```
