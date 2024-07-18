import streamlit as st
import random
from streamlit_calendar import calendar
import spacy
from dateutil.parser import parse
import datetime
from datetime import datetime , timedelta
import re
from llmhelper import *
import subprocess
import shlex


# Function to set a background image
def set_background(image_url):
    background_style = f"""
    <style>
    .stApp {{
        background-image: url({image_url});
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Function to customize sidebar appearance
def customize_sidebar():
    sidebar_style = """
    <style>
    .stSidebar {{
        background-color: #e0f2f1;
        color: #00695c;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    .stButton>button {{
        color: #ffffff;
        background-color: #00897b;
        border: none;
        border-radius: 5px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)

# Function to style text elements
def text_styling():
    text_style = """
    <style>
    h1, h2, h3, h4, h5, h6 {{
        color: #00796b;
        font-family: 'Arial';
    }}
    .stTextInput>label, .stSelectbox>label {{
        font-weight: bold;
        color: #004d40;
    }}
    .reportview-container .markdown-text-container {{
        font-family: 'Arial';
        font-size: 16px;
        color: #004d40;
    }}
    .stMarkdown {{
        color: #004d40;
    }}
    </style>
    """
    st.markdown(text_style, unsafe_allow_html=True)

import time

def get_tz():
    current_zone = time.tzname[time.daylight]
    return current_zone  

timezone = get_tz()

# Function to add custom HTML elements
def add_custom_elements(timezone):
    st.markdown(f"""
    <hr style='border-top: 4px dotted #005f56;'>
    <h2 style='text-align: center; color: #00897B;'>Welcome to Smart Calendar!</h2>
    <p style='text-align: center; color: #004D40;'>Plan your events efficiently and elegantly!</p>
    <p style='text-align: center; color: #004D40;'>Current Time Zone: {timezone}</p>
    """, unsafe_allow_html=True)

# Function to customize the calendar color scheme
def customize_calendar():
    calendar_style = """
    <style>
    /* Change the calendar header background and text color */
    .fc-header-toolbar {
        background-color: #00796b;
        color: #ffffff;
    }
    /* Change the day grid background and text color */
    .fc-daygrid-day {
        background-color: #e0f2f1;
        color: #00796b;
    }
    /* Change the background and text color for today */
    .fc-day-today {
        background-color: #004d40;
        color: #ffffff;
    }
    /* Change the event background color */
    .fc-event {
        background-color: #00897b;
        color: #ffffff;
    }
    </style>
    """
    st.markdown(calendar_style, unsafe_allow_html=True)

# Load the Spacy NLP model for date and time extraction
nlp = spacy.load("en_core_web_lg")

# Initialize session state for calendar events if not already done
if 'calendar_events' not in st.session_state:
    st.session_state['calendar_events'] = []

# Function to generate a random color for calendar events
def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


# Function to display the calendar
def display_calendar(events):
    st.write(calendar(events=events, options=calendar_options))


# Function to predict event type from text
def predict_event_type(text, model, tokenizer):
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    decoded_predictions = [model.config.id2label[pred] for pred in predictions]
    event_keyword_map = {
    'meet': 'Meeting',
    }
    for token, pred in zip(tokens, decoded_predictions):
        if token.lower() in event_keyword_map:
            return event_keyword_map[token.lower()]
    return "Event Scheduled"

# Function to add new events based on NLP parsed data, using event type for the title
#****************IMP**********************

def add_event():
    st.sidebar.subheader("Add New Event")

    llm = LLMHelper()
    event_input = st.sidebar.text_input("Describe your event", key="input")  # Define event_input here
    if event_input and st.sidebar.button("Schedule Event"):
        response = llm.generate_response(event_input)
        print(f"{response=}")
        summary = response["subject"]
        location = response["location"]
        start_date = response["date"][0]["start_date"]
        end_date = response["date"][0]["end_date"]
        start_time = response["start_time"]
        end_time = response["end_time"]
        listof_dates = response["list_of_dates"]

        try:
            event_start_date = start_date
            event_start_time = start_time
            print(f"{event_start_date=}, {event_start_time=}")  # Add debug statement
            event_start_datetime = datetime.strptime(start_date + " " + start_time, "%Y-%m-%d %H:%M:%S")

            event_start_date = event_start_datetime.date()
            event_start_time = event_start_datetime.time()
            event_start_datetime = datetime.combine(event_start_date, event_start_time)

            new_event = {
                "title": summary,
                "start": start_date,
                "end": end_date,
                "color": generate_random_color()
            }
            st.session_state['calendar_events'].append(new_event)
            st.sidebar.success(f"{summary} Scheduled!")

            # Run main.py using subprocess
            command = f"python3 main.py --event_input='{event_input}'"
            subprocess.run(shlex.split(command))

        finally:
            print("Execution Finished")



# Calendar options configuration
calendar_options = {
    "initialView": "dayGridMonth",  # Change initial view to dayGridMonth
    "headerToolbar": {
        "left": "prev,next today",
        "center": "title",
        "right": "dayGridMonth,timeGridWeek,timeGridDay,dayGridYear"  # Add dayGridYear button
    },
}

# Set up the page layout and styles
set_background('https://img.freepik.com/free-vector/gradient-mint-background_23-2150284420.jpg?w=2000&t=st=1713283548~exp=1713284148~hmac=33b075d34ffec40d28a7e55e30e721bb615e6d63a26ba60d56ab7bb4ddd052df')
customize_sidebar()
text_styling()
add_custom_elements(timezone)
customize_calendar()

# Login Page
if not st.session_state.get("logged_in", False):   # Add this condition
    st.title("Login")
    name = st.text_input("Name")
    email = st.text_input("Email")

    # Display the login button
    if st.button("Login"):
        if re.match(r"[^@]+@[^@]+\.[^@]+", email):
            st.session_state.logged_in = True
            st.session_state.user_info = {"name": name, "email": email}
        else:
            st.error("Please enter a valid email address.")

# If logged in, show the calendar and event input form
if st.session_state.get("logged_in", False):
    st.title("Smart Calendar")
    st.sidebar.title("Options")

    # Add new events form
    add_event()

    # Display the calendar with current events
    display_calendar(st.session_state['calendar_events'])
#----------------------------------------------------

# st.title("Smart Calendar")
# st.sidebar.title("Options")

# # Add new events form
# add_event()

# # Display the calendar with current events
# display_calendar(st.session_state['calendar_events'])

