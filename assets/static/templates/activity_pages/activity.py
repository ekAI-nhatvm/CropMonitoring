import datetime

import streamlit as st
from streamlit_calendar import calendar

st.set_page_config(page_title="Demo for streamlit-calendar", page_icon="📆")

with st.expander("**Information**", expanded=False):
    title_event1 = st.text_input("Event Title: ")
    start_date = st.date_input("Start Date: ")
    end_date = st.date_input("End Date: ")
    description = st.text_area("Description: ")
    color = st.selectbox("Choose Color:", ("red", "yellow", "orange", "green", "blue", "brown"))

    submit = st.button("Submit", type="primary")

mode = st.selectbox(
    "Choose Mode:",
    (
        "daygrid",
        "list",
        "multimonth",
    ),
)

if "events" not in st.session_state:
    st.session_state.events = [
        # ... Hardcoded data
    ]

if submit:
    new_event = {
        "title": title_event1,
        "color": color,
        "start": f"{start_date}",
        "end": f"{end_date}",
        "description": description,
    }
    st.session_state.events.append(new_event)

    st.snow()  # Successfully

calendar_options = {
    "editable": "true",
    "navLinks": "true",
    "selectable": "true",
    "eventContent": f'{title_event1}: {description}',
}

if mode == "daygrid":
    calendar_options.update({
        "headerToolbar": {
            "left": "today prev,next",
            "center": "title",
            "right": "dayGridDay,dayGridWeek,dayGridMonth",
        },
        "initialDate": datetime.date.today().isoformat(),
        "initialView": "dayGridMonth",
    })
elif mode == "list":
    calendar_options.update({
        "initialDate": datetime.date.today().isoformat(),
        "initialView": "listMonth",
    })
elif mode == "multimonth":
    calendar_options.update({
        "initialView": "multiMonthYear",
    })

state = calendar(
    events=st.session_state.get("events", []),
    options=calendar_options,
    custom_css="""
    .fc-event-past {
        opacity: 0.8;
    }
    .fc-event-time {
        font-style: italic;
    }
    .fc-event-title {
        font-weight: 700;
    }
    .fc-toolbar-title {
        font-size: 2rem;
    }
    """,
    key=mode,
)