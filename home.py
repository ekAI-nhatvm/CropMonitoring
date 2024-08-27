import streamlit as st


monitor = st.Page(
    "assets/static/templates/monitor_pages/monitoring.py",
    title="Bảng giám sát",
    icon=":material/healing:",
)

weather = st.Page(
    "assets/static/templates/monitor_pages/weather.py", title="Thời tiết", icon=":material/handyman:"
)

overview = st.Page(
    "assets/static/templates/overview_pages/overview.py",
    title="Phân tích tổng quan",
    icon=":material/person_add:",
)

leaderboard = st.Page("assets/static/templates/overview_pages/leaderboards.py", title="Bảng tổng quan", icon=":material/security:")

activity = st.Page("assets/static/templates/activity_pages/activity.py", title="Hoạt động")

team = st.Page("assets/static/templates/team_managements_pages/teammanagement.py", title="Thành viên")


monitor_page = [monitor, weather]
overview_page = [overview, leaderboard]
activity_page = [activity]
team_page = [team]


page_dict = {}
page_dict["monitor"] = monitor_page
page_dict["overview"] = overview_page
page_dict["activity"] = activity_page
page_dict["team"] = team_page

st.logo('assets/static/img/Logo.png')
pg = st.navigation(page_dict)
pg.run()

