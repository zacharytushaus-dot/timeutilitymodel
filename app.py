import streamlit as st

# Global config - this must be the first Streamlit command
st.set_page_config(page_title="Time Utility Model", layout="wide")

# Define your pages
# We point to your existing files. 
# url_path="client" ensures the URL looks like /client?run_id=...
client_page = st.Page("Client.py", title="Client View", url_path="client")
admin_page  = st.Page("Admin.py",  title="Admin Dashboard", url_path="admin", default=True)

# The Navigation Logic
# If we wanted to hide the Admin page from the sidebar entirely 
# when in client mode, we could add logic here, but this is the cleanest start.
pg = st.navigation({
    "Apps": [admin_page, client_page]
})

pg.run()