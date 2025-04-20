import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='Customer Personality Analysis', layout='wide')
st.title('**Customer Personality Analysis**')
st.markdown("""
<style>
    .main {
        background-color: #f4f6f8;
    }
</style>
""", unsafe_allow_html=True)

#Load data
df = pd.read_csv("marketing_campaign_cleaned.csv", sep="\t")
print(df.dtypes)
print(df.columns)
st.subheader('Raw Data Preview')
st.dataframe(df.head(), use_container_width=True)

#Feature Engineering
df['Total_Children'] = df['Kidhome'] + df['Teenhome']
df['Total_Mnt_Spent'] = (
    df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
    df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds'])
df['Total_Campaigns_Accepted'] = (
    df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] +
    df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response'])
from datetime import datetime
# Get current year
current_year = datetime.now().year
# Calculate Age
df['Age'] = current_year - df['Year_Birth']

#drops
df.drop('Year_Birth', axis=1, inplace=True)
df.drop('Kidhome', axis=1, inplace=True)
df.drop('Teenhome', axis=1, inplace=True)
df.drop('MntWines', axis=1, inplace=True)
df.drop('MntFruits', axis=1, inplace=True)
df.drop('MntMeatProducts', axis=1, inplace=True)
df.drop('MntFishProducts', axis=1, inplace=True)
df.drop('MntSweetProducts', axis=1, inplace=True)
df.drop('MntGoldProds', axis=1, inplace=True)
df.drop('AcceptedCmp1', axis=1, inplace=True)
df.drop('AcceptedCmp2', axis=1, inplace=True)
df.drop('AcceptedCmp3', axis=1, inplace=True)
df.drop('AcceptedCmp4', axis=1, inplace=True)
df.drop('AcceptedCmp5', axis=1, inplace=True)
df.drop('Dt_Customer', axis=1, inplace=True)


label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Marital_Status'] = label_encoder.fit_transform(df['Marital_Status'])

df.fillna(0, inplace=True)

features = df.select_dtypes(include=['int64', 'float64'])
df.drop('ID', inplace=True, axis=1)
#Normalizing
scaler = StandardScaler()
a_scaled = scaler.fit_transform(features)

#DBSCAN Settings:
st.sidebar.header('DBSCAN Settings:')
eps = st.sidebar.slider('Choose epsilon(eps)', 0.1, 5.0, 0.6)
min_samples = st.sidebar.slider('Min Samples', 1, 20, 5)
dbscan = DBSCAN(eps=eps, min_samples = min_samples)
labels = dbscan.fit_predict(a_scaled)

df['Cluster'] = labels
#Dimensionality Reduction
tsne = TSNE(n_components=3, perplexity=30)
X_tsne = tsne.fit_transform(a_scaled)
df['Dim1'] = X_tsne[:,0]
df['Dim2'] = X_tsne[:,1]

#Cluster Summary
summary = df.groupby('Cluster').agg({
    'Cluster': 'count',
    'Total_Mnt_Spent': 'mean',
    'Total_Campaigns_Accepted': 'mean',
    'Income': 'mean'
}).rename(columns={
    'Cluster': 'Customer Count',
    'Total_Mnt_Spent': 'Avg Spending',
    'Total_Campaigns_Accepted': 'Avg Campaign Rate',
    'Income': 'Avg Income'
})
summary['%Customer'] = round((summary['Customer Count'] / len(df))*100, 2)
st.dataframe(summary.reset_index(), use_container_width=True)

#Pie Chart
fig_pie = px.pie(summary, values='Customer Count', names=summary.index, title='Customers Distribution Per Cluster')
st.plotly_chart(fig_pie, use_container_width=True)

#t-SNE Scatter
st.subheader('t-SNE Clustering Visualization')
fig_tsne = px.scatter(df, x='Dim1', y='Dim2', color='Cluster', color_continuous_scale='viridis', title='t-SNE Clustering Projection')
st.plotly_chart(fig_tsne)

#Bar Chart
fig_bar = px.bar(summary.reset_index(), x='Cluster', y=['Avg Income', 'Avg Spending', 'Avg Campaign Rate'], title='Avg Income, Avg Spending & Avg Campaign Rate', barmode='group')
st.plotly_chart(fig_bar, use_container_width=True)

#Final Data Review
st.subheader('Final Data Review')
st.dataframe(df, use_container_width=True)
st.success('Dashboard Loaded Successfully!!')