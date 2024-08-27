import streamlit as st
import leafmap.foliumap as lf
from streamlit_folium import st_folium
import pandas as pd
import requests
import json

st.set_page_config(page_title="Giám sát trang trại", layout="wide")


if 'farms' not in st.session_state:
    st.session_state.farms = {
        'Trang trại 1': {
            'area_location': [[21.037083, 105.830956], [21.030033, 105.831127], [21.029872, 105.84383],
                              [21.036602, 105.844774], [21.037083, 105.830956]],
            'data_ndvi': [1, 1, 1],
            'data_msavi': [1, 1, 1]
        },
        'Trang trại 2': {
            'area_location': [[21.04269, 105.794392], [21.041889, 105.800657], [21.03508, 105.790272],
                              [21.037723, 105.784864], [21.04269, 105.794392]],
            'data_ndvi': [2, 2, 2],
            'data_msavi': [2, 2, 2]
        }
    }

st.title("Giám sát trang trại")

row1_col1, row1_col2 = st.columns([3, 1])

with st.sidebar:
    st.title("Vụ mùa 2024")
    st.header("Thông tin chi tiết")
    st.text("Ngày bắt đầu: 2023-01-01")
    st.text("Ngày kết thúc: 2023-01-08")

    st.header("Chọn trang trại")

    selected_farm = st.selectbox("Chọn trang trại", list(st.session_state.farms.keys()), key='farm_selection')

    if selected_farm:
        st.write(f"Thông tin trang trại {selected_farm}:")
        st.write(st.session_state.farms[selected_farm])

    st.subheader("Thêm trang trại")
    new_farm_name = st.text_input("Tên trang trại")

    if st.button("Thêm trang trại từ bản đồ"):
        if new_farm_name and 'new_farm_area' in st.session_state:
            st.session_state.farms[new_farm_name] = {
                'area_location': st.session_state.new_farm_area,
                'data_ndvi': [],
                'data_msavi': []
            }
            st.success(f"Trang trại '{new_farm_name}' đã được thêm thành công!")
            st.experimental_rerun()
        else:
            st.error("Chưa có tên trang trại hoặc chưa vẽ khu vực!")


def get_weather_data(lat, lon):
    response = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid=54fa305a99c9bd3d195dc7ce4eccb20b")
    return response.json()


def create_and_display_map():
    row1_col1, row1_col2 = st.columns([3, 1])
    height = 600
    width = 800

    with row1_col1:
        m = lf.Map(center=[21.037083, 105.830956], zoom=14, draw_export=True)
        m.add_basemap('SATELLITE')

        map_data = st_folium(m, height=height, width=width)

        if map_data and map_data.get('last_active_drawing'):
            st.session_state.new_farm_area = map_data['last_active_drawing']['geometry']['coordinates'][0]
            st.write("Tọa độ trang trại mới:")
            st.json(st.session_state.new_farm_area)

    if selected_farm in st.session_state.farms:
        with row1_col2:
            latitude = st.session_state.farms[selected_farm]['area_location'][0][0]
            longitude = st.session_state.farms[selected_farm]['area_location'][0][1]
            weather_data = get_weather_data(latitude, longitude)

            st.subheader('Thời tiết hiện tại')
            st.write(f"Nhiệt độ: {round(weather_data['main']['temp'] - 273.15, 2)}°C")
            st.write(f"Nhiệt độ thấp nhất: {round(weather_data['main']['temp_min'] - 273.15, 2)}°C")
            st.write(f"Nhiệt độ cao nhất: {round(weather_data['main']['temp_max'] - 273.15, 2)}°C")
            st.write(f"Áp suất: {weather_data['main']['pressure']} hPa")
            st.write(f"Độ ẩm: {weather_data['main']['humidity']}%")
            st.write(f"Tốc độ gió: {weather_data['wind']['speed']} m/s")
            st.write(f"Hướng gió: {weather_data['wind']['deg']}°")
            st.write(f"Trạng thái thời tiết: {weather_data['weather'][0]['description'].capitalize()}")
            st.write(f"Mây: {weather_data['clouds']['all']}%")


create_and_display_map()
