import streamlit as st
import pandas as pd

st.set_page_config(page_title="Employee Manager", layout="wide", page_icon=':sunglasses:')

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['ID', 'Name', 'Age', 'Position'])

def add_employee(id, name, age, position):
    if id in st.session_state.df['ID'].values:
        st.error('ID already exists')
    else:
        new_employee = {'ID': id, 'Name': name, 'Age': age, 'Position': position}
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_employee])], ignore_index=True)
        st.success(f"Added employee: {name}")

def delete_employee(id):
    if id not in st.session_state.df['ID'].values:
        st.error('ID does not exist')
    else:
        st.session_state.df = st.session_state.df[st.session_state.df['ID'] != id]
        st.success(f"Deleted employee with ID: {id}")
def update_employee(id, name, age, position):
    st.session_state.df.loc[st.session_state.df['ID'] == id, ['Name', 'Age', 'Position']] = [name, age, position]

st.title("Employee Management System")

menu = ["Add", "Delete", "Update", "View"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add":
    st.subheader("Add Employee")
    id = st.text_input("ID")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=18, max_value=65)
    position = st.text_input("Position")
    if st.button("Add Employee"):
        add_employee(id, name, age, position)

elif choice == "Delete":
    st.subheader("Delete Employee")
    id = st.text_input("ID")
    if st.button("Delete Employee"):
        delete_employee(id)

elif choice == "Update":
    st.subheader("Update Employee")
    id = st.text_input("ID")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=18, max_value=65)
    position = st.text_input("Position")
    if st.button("Update Employee"):
        update_employee(id, name, age, position)
        st.success(f"Updated employee with ID: {id}")

elif choice == "View":
    st.subheader("View Employees")
    st.dataframe(st.session_state.df)

if st.checkbox("Show Employee Data"):
    st.dataframe(st.session_state.df)

