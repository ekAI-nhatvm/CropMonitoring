import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Sample data
crop_data = {
    "Tên": ["Lúa", "Ngô"],
    "Trang trại": [1, 1],
    "Diện tích": ["30.6 ha", "33 ha"],
    "Cảnh báo": ["N/A", "N/A"]
}

area_distribution = {
    "Cây trồng": ["Lúa", "Ngô"],
    "Diện tích (ha)": [30.6, 33]
}

ndvi_data = {
    "Weeks": list(range(1, 24)),
    "NDVI": [0.65, 0.66, 0.67, 0.68, 0.69, 0.68, 0.67, 0.68, 0.69, 0.68, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.71, 0.70, 0.69, 0.70, 0.71, 0.70, 0.69]
}

# Convert data to pandas DataFrame
df_crop = pd.DataFrame(crop_data)
df_area = pd.DataFrame(area_distribution)
df_ndvi = pd.DataFrame(ndvi_data)

# Set page layout
st.set_page_config(layout="wide")

# Dashboard title
st.title("Bảng giám sát tổng quan")

# Crop details table
with st.container():
    st.subheader("Cây trồng")
    st.table(df_crop)

# Area distribution pie chart
with st.container():
    st.subheader("Tỉ lệ diện tích đất trồng")
    fig_pie = px.pie(df_area, names="Cây trồng", values="Diện tích (ha)", color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig_pie)

# NDVI monitoring line chart
with st.container():
    st.subheader("Giám sát chỉ số NDVI (Hàng tuần)")
    fig_ndvi = go.Figure()
    fig_ndvi.add_trace(go.Scatter(x=df_ndvi["Weeks"], y=df_ndvi["NDVI"], mode='lines+markers', name='NDVI'))
    fig_ndvi.update_layout(xaxis_title="Weeks", yaxis_title="NDVI", template="plotly_dark")
    st.plotly_chart(fig_ndvi)
