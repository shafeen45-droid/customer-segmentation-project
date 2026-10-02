# 👥 Customer Segmentation Analytics Dashboard

A cloud-deployed machine learning application that segments an operational customer base into actionable behavioral profiles using **K-Means Clustering** via **scikit-learn** and dynamic **Streamlit UI elements**.

## 📊 Project Objectives & Deliverables
* **Algorithmic Profiling:** Implements automated data scaling (`StandardScaler`) and unsupervised learning to cluster consumers based on age, income brackets, and historical spending data.
* **Interactive 3D Visualizations:** Houses an adaptive, multi-dimensional Plotly 3D engine enabling corporate reviewers to isolate specific consumer segments dynamically.
* **Granular Configuration Controls:** Features real-time parameter tweaking sliders to adjust customer age ranges or modify the model's target number of segments (K) on the fly.

## 🛠️ System Frameworks Used
* **Backend Pipeline:** Python 3, Pandas, scikit-learn
* **Data Visualization Canvas:** Plotly Express
* **Web UI Framework:** Streamlit Cloud Architecture

## 🚀 Repository Blueprint
* `segmentation_app.py` - Core application layer script managing calculations, visual bindings, and the ML runtime engine.
* `customer_data.csv` - Underlying structured consumer data repository.
* `requirements.txt` - Deployment dependency manifest cataloging all needed workspace framework installations.
