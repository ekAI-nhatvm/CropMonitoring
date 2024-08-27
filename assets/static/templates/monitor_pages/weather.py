import streamlit as st
from streamlit_folium import st_folium, folium
import leafmap.foliumap as lf
# from folium.plugins import HeatMap
# from folium.plugins import HeatMapWithTime
import pandas as pd 
import altair as alt
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Thời tiết", layout="wide")
st.header("Thời tiết")

ndvi_data1 = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'],
    'ndvi': [0.6, 0.65, 0.7, 0.8, 0.5, 0.3, 0.35, 0.9],
    'temp': [30, 28, 29, 31, 33, 35,34, 30],
    'rainfall': [5, 10, 7, 12,6,3,15,4]
}

ndvi_data2 = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'],
    'ndvi': [0.1, 0.45, 0.34, 0.8, 0.7, 0.75, 0.9, 0.6],
    'temp': [30.5, 28.1, 29.6, 31.1, 33.9, 35.1,34.4, 30.6],
    'rainfall': [1, 3, 2, 12,9,6,7,3]
}

# Thiết lập tiêu đề và giao diện

selected_info_farm = []
row1_col1, row1_col2 = st.columns([3, 1])
# Sidebar - bảng điều khiển
with st.sidebar:
    st.title("Vụ mùa 2024")
    st.header("Thông tin chi tiết")
    st.text("Ngày bắt đầu: 2023-01-01")
    st.text("Ngày kết thúc: 2023-01-08")
    # Chọn trang trại
    st.header("Chọn trang trại")
    farms = {
        'Trang trại 1': {'area_location': [[21.037083, 105.830956],[21.030033, 105.831127],[21.029872, 105.84383],[21.036602, 105.844774],[21.037083, 105.830956]], 'data_ndvi':ndvi_data1, 'data_msavi' : [1,1,1]},
        'Trang trại 2': {'area_location': [[21.04269, 105.794392],[21.041889, 105.800657],[21.03508, 105.790272],[21.037723, 105.784864],[21.04269, 105.794392]], 'data_ndvi':ndvi_data2, 'data_msavi' : [2,2,2]},
    }

    #[21.037083,105.830956],[21.030033,105.831127],[21.029872,105.84383],[21.036602,105.844774],[21.037083,105.830956]
    selected_farm = st.selectbox("Chọn trang trại", list(farms.keys()))
    for name, info in farms.items():
        if name == selected_farm:
            selected_info_farm.append([name,info])
            print(selected_info_farm)

    # Thêm trang trại mới
    st.subheader("Thêm trang trại")
    new_farm_name = st.text_input("Tên trang trại")
    polygon_field = st.text_input("Khoanh vùng trang trại")

    if st.button("Thêm trang trại"):
        st.write(f"Thêm trang trại: {new_farm_name}, {polygon_field}")





# Map
def app():
    row1_col1, row1_col2 = st.columns([3, 1])
    width = 800
    height = 600
    
    with row1_col2:

        def get_weather_data(lat, lon):
            response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid=54fa305a99c9bd3d195dc7ce4eccb20b")
            return response.json()

            # Lấy dữ liệu dự báo thời tiết
        latitude = selected_info_farm[0][1]['area_location'][0][0]
        longitude = selected_info_farm[0][1]['area_location'][0][1]
        weather_data = get_weather_data(latitude, longitude)
        print(weather_data)
        st.subheader('Thời tiết hiện tại')
        st.write(f"Nhiệt độ: {round(weather_data['main']['temp'] - 273.15,2)}°C")
        st.write(f"Nhiệt độ thấp nhất: {round(weather_data['main']['temp_min'] - 273.15,2)}°C")
        st.write(f"Nhiệt độ cao nhất: {round(weather_data['main']['temp_max'] - 273.15,2)}°C")
        st.write(f"Áp suất: {weather_data['main']['pressure']} hPa")
        st.write(f"Độ ẩm: {weather_data['main']['humidity']}%")
        st.write(f"Tầm nhìn: {weather_data['visibility']} m")
        st.write(f"Tốc độ gió: {weather_data['wind']['speed']} m/s")
        st.write(f"Hướng gió: {weather_data['wind']['deg']}°")
        st.write(f"Trạng thái thời tiết: {weather_data['weather'][0]['description'].capitalize()}")
        st.write(f"Mây: {weather_data['clouds']['all']}%")
        

        with row1_col1:
            # chart
            # Dữ liệu NDVI theo ngày (ví dụ)
            #luong mua va ndvi
            st.title("NDVI")
            ndvi_df = pd.DataFrame(selected_info_farm[0][1]['data_ndvi'])
            selected_index = st.selectbox("Chọn chỉ số", list(['NDVI', 'MSAVI']))
            
            base = alt.Chart(ndvi_df).encode(
                alt.X('date:T', axis=alt.Axis(title='Ngày'))
            )

            bar = base.mark_bar(color='#FFDB58').encode(
                alt.Y('rainfall:Q', axis=alt.Axis(title='Lượng mưa (mm)'))
            )

            line = base.mark_line(color='#00FFB9', point=True).encode(
                alt.Y('ndvi:Q', axis=alt.Axis(title='Chỉ số NDVI'))
            )

            combined_chart = alt.layer(bar, line).resolve_scale(
                y='independent'
            ).properties(
                title='Lượng mưa và Chỉ số NDVI theo ngày'
            ).interactive()
            st.altair_chart(combined_chart, use_container_width=True)
            # nhiet do va ndvi
            tan = alt.Chart(ndvi_df).encode(
                alt.X('date:T', axis=alt.Axis(title='Ngày'))
            )

            bar_tan = tan.mark_line(color='#FFDB58', point=True).encode(
                alt.Y('temp:Q', axis=alt.Axis(title='Nhiệt độ (C)'))
            )

            line_tan = tan.mark_line(color='#00FFB9', point=True).encode(
                alt.Y('ndvi:Q', axis=alt.Axis(title='Chỉ số NDVI'))
            )

            combined_chart_tan = alt.layer(bar_tan, line_tan).resolve_scale(
                y='independent'
            ).properties(
                title='Nhiệt độ và Chỉ số NDVI theo ngày'
            ).interactive()
            # Hiển thị biểu đồ
            st.altair_chart(combined_chart_tan, use_container_width=True)
            # latitude = selected_info_farm[0][1]['area_location'][0][0]
            # longitude = selected_info_farm[0][1]['area_location'][0][1]
            selected_info_farm.clear()  
            
            
            # Thong tin thoi tiet va du bao 
            def fetch_weather_data(lat, lon):
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"  # Replace with the actual URL
                response = requests.get(url)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    st.error("Failed to fetch data from the API")
                    return None
            st.title("Thời tiết")
            data = fetch_weather_data(latitude, longitude)
            if data is None:
                return
            
            # Convert time data to pandas datetime
            time = pd.to_datetime(data["hourly"]["time"])
            temperature_2m = data["hourly"]["temperature_2m"]
            relative_humidity_2m = data["hourly"]["relative_humidity_2m"]
            wind_speed_10m = data["hourly"]["wind_speed_10m"]

            # Create dataframes
            df_temperature = pd.DataFrame({
                "Time": time,
                "Temperature (°C)": temperature_2m
            })
            
            df_humidity = pd.DataFrame({
                "Time": time,
                "Relative Humidity (%)": relative_humidity_2m
            })
            
            df_wind_speed = pd.DataFrame({
                "Time": time,
                "Wind Speed (km/h)": wind_speed_10m
            })

            df_combined = pd.DataFrame({
                "Time": time,
                "Temperature (°C)": temperature_2m,
                "Relative Humidity (%)": relative_humidity_2m,
                "Wind Speed (km/h)": wind_speed_10m
            })
            # table 
            # Resample to daily means
            df_daily = df_combined.resample('D', on='Time').mean().reset_index()
            # Display the daily summary
            st.subheader("Thời tiết trung bình hàng ngày")
            st.dataframe(df_daily)
            # Create a dropdown menu for selecting a day
            selected_date = st.selectbox("Select a day", df_daily["Time"].dt.strftime('%Y-%m-%d'))
            
            # Filter the detailed data for the selected day
            selected_day_data = df_combined[df_combined["Time"].dt.strftime('%Y-%m-%d') == selected_date]
            # Display detailed data for the selected day
            if not selected_day_data.empty:
                st.subheader(f"Chi tiết thời tiết cho ngày {selected_date}")
                st.dataframe(selected_day_data)
            else:
                st.warning(f"Không có dữ liệu cho ngày {selected_date}")

            # line chart
            current_time = datetime.now()
            # Plot temperature data
            st.subheader("Nhiệt độ theo thời gian")
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=df_temperature["Time"], y=df_temperature["Temperature (°C)"],
                                        mode='lines', name='Temperature'))
            fig_temp.add_vline(x=current_time, line=dict(color='red', width=2, dash='dash'), name='Current Time')
            fig_temp.add_annotation(x=current_time, y=max(df_temperature["Temperature (°C)"]),
                            text="Current Time", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_temp)

            # Plot humidity data
            st.subheader("Độ ẩm theo thời gian")
            fig_humidity = go.Figure()
            fig_humidity.add_trace(go.Scatter(x=df_humidity["Time"], y=df_humidity["Relative Humidity (%)"],
                                            mode='lines', name='Relative Humidity'))
            fig_humidity.add_vline(x=current_time, line=dict(color='red', width=2, dash='dash'), name='Current Time')
            fig_humidity.add_annotation(x=current_time, y=max(df_humidity["Relative Humidity (%)"]),
                                text="Current Time", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_humidity)

            # Plot wind speed data
            st.subheader("Tốc độ gió theo thời gian")
            fig_wind_speed = go.Figure()
            fig_wind_speed.add_trace(go.Scatter(x=df_wind_speed["Time"], y=df_wind_speed["Wind Speed (km/h)"],
                                                mode='lines', name='Wind Speed'))
            fig_wind_speed.add_vline(x=current_time, line=dict(color='red', width=2, dash='dash'), name='Current Time')
            fig_wind_speed.add_annotation(x=current_time, y=max(df_wind_speed["Wind Speed (km/h)"]),
                                  text="Current Time", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_wind_speed)
app()
