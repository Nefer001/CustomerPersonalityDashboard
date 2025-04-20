# 🧠 Customer Personality Dashboard

This interactive dashboard helps visualize and analyze customer personality insights using unsupervised machine learning techniques like DBSCAN and t-SNE.

It allows businesses and marketers to segment customers based on behavior, spending habits, and lifestyle — enabling more targeted marketing and strategic decision-making.

---

## 🚀 Live Demo

👉 [Click here to view the live dashboard on Streamlit](https://customerpersonalitydashboard.streamlit.app/)

---

## 📊 Features

- Customer segmentation using DBSCAN Clustering
- Dimensionality reduction with t-SNE
- Interactive visualizations and filters
- Clean, responsive UI using Streamlit

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn** (DBSCAN, t-SNE)
- **Matplotlib & Seaborn** (for EDA)
- **Streamlit** (for dashboard)

---

## ⚙️ How to Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/Nefer001/CustomerPersonalityDashboard.git
cd CustomerPersonalityDashboard

Create virtual environment (optional but recommended)
bash:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install requirements
bash:
pip install -r requirements.txt

Run Streamlit app
bash:
streamlit run app.py
