import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration layout
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("👥 Customer Segmentation Analytics Dashboard")
st.markdown("Analyze purchase behaviors and group customers using **Machine Learning (K-Means Clustering)**.")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("customer_data.csv")
        return df
    except FileNotFoundError:
        st.error("Error: 'customer_data.csv' file not found. Please ensure it is uploaded in the same repository.")
        return None

df = load_data()

if df is not None:
    # Sidebar Filters
    st.sidebar.header("🔧 Model Configuration")
    num_clusters = st.sidebar.slider("Select Number of Segments (K)", min_value=2, max_value=6, value=4)
    
    st.sidebar.header("📊 Demographic Slicers")
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_filter = st.sidebar.slider("Filter by Age Range", min_age, max_age, (min_age, max_age))
    
    # Filter dataset based on sidebar options
    filtered_df = df[(df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1])].copy()

    # Perform K-Means Clustering on the filtered dataset
    features = ['Age', 'Annual_Income', 'Spending_Score']
    X = filtered_df[features]
    
    if len(X) >= num_clusters:
        # Standardize features for clean clustering calculations
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        filtered_df['Segment_ID'] = kmeans.fit_predict(X_scaled)
        
        # Map structural IDs to readable descriptive segment profile tags
        cluster_labels = {
            0: "Budget Shoppers",
            1: "High Earners (Conservative)",
            2: "Target Elite (High Spenders)",
            3: "Young Dynamic Shoppers",
            4: "Stagnant/At-Risk Profiles",
            5: "Occasional Buyers"
        }
        filtered_df['Segment Name'] = filtered_df['Segment_ID'].map(cluster_labels)

        # Main Layout: 3 KPI Column Blocks
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Total Sample Customers", f"{len(filtered_df)}")
        with kpi2:
            st.metric("Average Annual Income", f"${filtered_df['Annual_Income'].mean():.1f}K")
        with kpi3:
            st.metric("Average Spending Score (1-100)", f"{filtered_df['Spending_Score'].mean():.1f}")
            
        st.markdown("---")

        # Visualizations Columns Grid Section
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("🪐 3D Behavior Segmentation Cluster Mapping")
            fig_3d = px.scatter_3d(
                filtered_df, 
                x='Age', 
                y='Annual_Income', 
                z='Spending_Score',
                color='Segment Name',
                hover_data=['CustomerID', 'Total_Purchases'],
                labels={'Annual_Income': 'Annual Income ($K)', 'Spending_Score': 'Spending Score (1-100)'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=550)
            st.plotly_chart(fig_3d, use_container_width=True)

        with col2:
            st.subheader("📈 Purchases vs Spending Score")
            fig_2d = px.scatter(
                filtered_df,
                x='Spending_Score',
                y='Total_Purchases',
                color='Segment Name',
                size='Annual_Income',
                labels={'Spending_Score': 'Spending Score (1-100)', 'Total_Purchases': 'Number of Purchases'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_2d.update_layout(height=550)
            st.plotly_chart(fig_2d, use_container_width=True)

        st.markdown("---")
        
        # Segment Analysis Summary Profile Table Area
        st.subheader("📋 Machine Learning Segment Statistical Profile Matrices")
        summary_table = filtered_df.groupby('Segment Name')[['Age', 'Annual_Income', 'Spending_Score', 'Total_Purchases']].mean()
        summary_table.columns = ['Avg Age', 'Avg Income ($K)', 'Avg Spending Score', 'Avg Total Purchases']
        st.dataframe(summary_table.style.format("{:.1f}").background_gradient(cmap='Blues'), use_container_width=True)
        
    else:
        st.warning("Not enough data points available in the current age filter configuration to accurately train the specified number of clusters.")
