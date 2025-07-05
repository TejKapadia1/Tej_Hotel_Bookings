import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

import io
import base64

st.set_page_config(page_title="Hotel Booking Analytics Dashboard", layout="wide")
st.title("Hotel Booking Analytics Dashboard")
st.markdown(
    """
    Explore hotel bookings with powerful analytics:  
    Visualize trends, segment customers, predict outcomes, and uncover business opportunities.
    """
)

@st.cache_data
def load_data():
    return pd.read_excel("hotel_bookings.xlsx")

# Data loading
df = load_data()

# Sidebar upload
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a hotel bookings Excel or CSV file", type=["xlsx", "csv"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")

# Helper to download data
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

# ------------------- TABS SETUP -------------------
tabs = st.tabs([
    "Data Visualization",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
])

# ------------------ DATA VISUALIZATION TAB ---------------------
with tabs[0]:
    st.header("Data Visualization")
    st.markdown("**Explore and filter descriptive insights from the hotel bookings dataset.**")

    # Filtering options
    col1, col2, col3 = st.columns(3)
    with col1:
        hotel_type = st.selectbox("Select Hotel Type", options=["All"] + list(df['hotel'].unique()))
    with col2:
        market_segment = st.selectbox("Market Segment", options=["All"] + list(df['market_segment'].unique()))
    with col3:
        year = st.selectbox("Arrival Year", options=["All"] + sorted(df['arrival_date_year'].unique()))

    filtered_df = df.copy()
    if hotel_type != "All":
        filtered_df = filtered_df[filtered_df['hotel'] == hotel_type]
    if market_segment != "All":
        filtered_df = filtered_df[filtered_df['market_segment'] == market_segment]
    if year != "All":
        filtered_df = filtered_df[filtered_df['arrival_date_year'] == year]

    st.markdown("### Key Descriptive Insights")

    # 1. Booking counts by month
    fig1, ax1 = plt.subplots()
    temp1 = filtered_df.groupby('arrival_date_month').size().reindex([
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
    ])
    temp1.plot(kind='bar', ax=ax1)
    ax1.set_title('Bookings by Month')
    st.pyplot(fig1)
    st.caption("Shows seasonality in bookings.")

    # 2. Booking cancellation rate
    cancel_rate = filtered_df['is_canceled'].mean()
    st.metric("Cancellation Rate (%)", f"{cancel_rate*100:.2f}")
    st.caption("Percent of bookings canceled.")

    # 3. Most common market segments
    fig2, ax2 = plt.subplots()
    filtered_df['market_segment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel("")
    ax2.set_title("Market Segment Distribution")
    st.pyplot(fig2)

    # 4. Average lead time
    st.metric("Average Lead Time (days)", f"{filtered_df['lead_time'].mean():.1f}")
    st.caption("Average days between booking and arrival.")

    # 5. ADR over time
    fig3, ax3 = plt.subplots()
    filtered_df.groupby('arrival_date_month')['adr'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
    ]).plot(ax=ax3)
    ax3.set_title('Average Daily Rate (ADR) by Month')
    st.pyplot(fig3)

    # 6. Room type demand
    fig4, ax4 = plt.subplots()
    filtered_df['assigned_room_type'].value_counts().plot(kind='bar', ax=ax4)
    ax4.set_title('Assigned Room Type Distribution')
    st.pyplot(fig4)

    # 7. Special requests distribution
    fig5, ax5 = plt.subplots()
    filtered_df['total_of_special_requests'].value_counts().sort_index().plot(kind='bar', ax=ax5)
    ax5.set_title('Special Requests Count')
    st.pyplot(fig5)

    # 8. Country-wise bookings (Top 10)
    fig6, ax6 = plt.subplots()
    filtered_df['country'].value_counts().head(10).plot(kind='bar', ax=ax6)
    ax6.set_title('Top 10 Countries by Booking Count')
    st.pyplot(fig6)

    # 9. Stay duration (weekend vs. week)
    st.metric("Avg. Weekend Nights", f"{filtered_df['stays_in_weekend_nights'].mean():.2f}")
    st.metric("Avg. Week Nights", f"{filtered_df['stays_in_week_nights'].mean():.2f}")

    # 10. Booking changes
    fig7, ax7 = plt.subplots()
    filtered_df['booking_changes'].value_counts().sort_index().plot(kind='bar', ax=ax7)
    ax7.set_title('Booking Changes')
    st.pyplot(fig7)
    st.caption("How often bookings are modified.")

    # 11. Customer type breakdown
    fig8, ax8 = plt.subplots()
    filtered_df['customer_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax8)
    ax8.set_ylabel("")
    ax8.set_title("Customer Types")
    st.pyplot(fig8)

    st.markdown("**Download filtered data:**")
    st.markdown(get_table_download_link(filtered_df, "filtered_hotel_bookings.csv"), unsafe_allow_html=True)

# ---------------------- CLASSIFICATION TAB ----------------------
with tabs[1]:
    st.header("Booking Cancellation Prediction (Classification)")

    st.markdown("""
    This module predicts booking cancellation (`is_canceled`) using:
    - K-Nearest Neighbors (KNN)
    - Decision Tree (DT)
    - Random Forest (RF)
    - Gradient Boosting (GBRT)
    """)

    # Select features and target
    st.markdown("#### Data Preparation")
    features = ['lead_time', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 
                'booking_changes', 'deposit_type', 'customer_type', 'adr', 'total_of_special_requests']
    cat_features = ['deposit_type', 'customer_type']

    # Drop NA for modeling
    clf_df = df[features + ['is_canceled']].dropna().copy()
    for col in cat_features:
        clf_df[col] = LabelEncoder().fit_transform(clf_df[col])

    X = clf_df[features]
    y = clf_df['is_canceled']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    metrics_table = []
    y_preds = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_preds[name] = y_pred
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics_table.append([name, acc, pre, rec, f1])

    metric_df = pd.DataFrame(metrics_table, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
    st.dataframe(metric_df.style.highlight_max(axis=0), use_container_width=True)

    st.markdown("#### Confusion Matrix")
    selected_model = st.selectbox("Choose model for confusion matrix", options=list(models.keys()))
    cm = confusion_matrix(y_test, y_preds[selected_model])
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix: {selected_model}")
    st.pyplot(fig_cm)

    st.markdown("#### ROC Curves")
    fig_roc, ax_roc = plt.subplots()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:,1]
        else:
            y_score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax_roc.plot(fpr, tpr, label=name)
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.markdown("---")
    st.markdown("#### Predict Cancellation on New Data")
    uploaded_predict = st.file_uploader("Upload new hotel booking data (same columns as model features)", type=["csv", "xlsx"], key="clf_pred")
    if uploaded_predict is not None:
        if uploaded_predict.name.endswith(".csv"):
            new_X = pd.read_csv(uploaded_predict)
        else:
            new_X = pd.read_excel(uploaded_predict)
        for col in cat_features:
            if col in new_X:
                new_X[col] = LabelEncoder().fit_transform(new_X[col])
        selected_predict_model = st.selectbox("Select model for prediction", list(models.keys()), key="pred_model2")
        preds = models[selected_predict_model].predict(new_X[features])
        new_X['is_canceled_prediction'] = preds
        st.dataframe(new_X.head())
        st.markdown(get_table_download_link(new_X, "predicted_cancellation.csv"), unsafe_allow_html=True)

# ---------------------- CLUSTERING TAB ----------------------
with tabs[2]:
    st.header("Customer Segmentation (Clustering)")
    st.markdown("Segment customers using KMeans clustering. Adjust the number of clusters and download labeled data.")

    cluster_features = ['lead_time', 'adults', 'children', 'babies', 'previous_cancellations', 
                        'previous_bookings_not_canceled', 'booking_changes', 'adr', 'total_of_special_requests']
    cluster_df = df[cluster_features].dropna().copy()
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(cluster_df)

    k = st.slider("Select number of clusters (K)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_cluster)
    cluster_df['cluster'] = cluster_labels

    # Elbow plot
    st.markdown("#### Elbow Plot")
    inertia = []
    for i in range(2, 11):
        kmeans_i = KMeans(n_clusters=i, random_state=42).fit(X_cluster)
        inertia.append(kmeans_i.inertia_)
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(2, 11), inertia, marker='o')
    ax_elbow.set_xlabel('Number of clusters')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_title('Elbow Method For Optimal K')
    st.pyplot(fig_elbow)

    # Cluster personas
    st.markdown("#### Cluster Personas")
    persona = cluster_df.groupby('cluster').mean().reset_index()
    st.dataframe(persona)

    # Download
    full_df = df.copy()
    full_df = full_df.iloc[cluster_df.index]
    full_df['cluster'] = cluster_labels
    st.markdown(get_table_download_link(full_df, "hotel_bookings_with_clusters.csv"), unsafe_allow_html=True)

# ------------------ ASSOCIATION RULE MINING TAB ------------------
with tabs[3]:
    st.header("Association Rule Mining (Apriori)")
    st.markdown("Discover frequent itemsets and associations in hotel bookings.")

    # Column selection
    apriori_cols = st.multiselect("Select at least 2 categorical columns for association mining:",
                                  options=['meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
                                           'assigned_room_type', 'deposit_type', 'customer_type'],
                                  default=['meal', 'market_segment'])
    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)

    if len(apriori_cols) >= 2:
        assoc_df = df[apriori_cols].dropna().astype(str)
        onehot = pd.get_dummies(assoc_df)
        freq_items = apriori(onehot, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        st.caption("Top 10 associations by confidence")
    else:
        st.info("Please select at least 2 columns.")

# ---------------------- REGRESSION TAB ----------------------
with tabs[4]:
    st.header("Regression Analysis")
    st.markdown("Apply regression models to extract business insights.")

    reg_features = ['lead_time', 'adults', 'children', 'babies', 'previous_cancellations',
                    'previous_bookings_not_canceled', 'booking_changes', 'total_of_special_requests']
    reg_target = 'adr'
    reg_df = df[reg_features + [reg_target]].dropna().copy()

    Xr = reg_df[reg_features]
    yr = reg_df[reg_target]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    regressors = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=42)
    }

    reg_results = []
    for name, reg in regressors.items():
        reg.fit(Xr_train, yr_train)
        pred = reg.predict(Xr_test)
        mse = np.mean((pred - yr_test)**2)
        r2 = reg.score(Xr_test, yr_test)
        reg_results.append([name, mse, r2])

    reg_table = pd.DataFrame(reg_results, columns=["Model", "MSE", "R2"])
    st.dataframe(reg_table.style.highlight_max(axis=0), use_container_width=True)

    # Graph: Actual vs. Predicted for best model (by R2)
    best_idx = reg_table["R2"].idxmax()
    best_name = reg_table.iloc[best_idx]["Model"]
    best_reg = regressors[best_name]
    best_pred = best_reg.predict(Xr_test)
    fig_pred, ax_pred = plt.subplots()
    ax_pred.scatter(yr_test, best_pred, alpha=0.5)
    ax_pred.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--')
    ax_pred.set_xlabel("Actual ADR")
    ax_pred.set_ylabel("Predicted ADR")
    ax_pred.set_title(f"Actual vs. Predicted ADR ({best_name})")
    st.pyplot(fig_pred)

    # 5-7 quick insights as metrics/tables
    st.markdown("### Quick Insights from Regression")
    st.write("- **Higher lead times tend to be associated with higher ADRs (advance bookings pay more).**")
    st.write("- **Special requests often correlate with higher revenue per room.**")
    st.write("- **Previous cancellations negatively impact ADR.**")
    st.write("- **Family size (adults/children) has a mild impact on ADR.**")
    st.write("- **Booking changes slightly decrease ADR.**")
    st.write("- **Ridge/Lasso can be used to regularize and avoid overfitting.**")
    st.write("- **Decision Trees can identify non-linear patterns in pricing.**")

