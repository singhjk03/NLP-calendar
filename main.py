import os
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import pytz
from llmhelper import *

load_dotenv()

import argparse

#  event_start_time='16:00:00'
# start_date='2024-07-12'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_input", type=str, help="Input for event description")
    args = parser.parse_args()

    event_input = args.event_input
    print(f"Event input received: {event_input}")

    # Rest of your main.py code...

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    creds = None
    # Check for existing token file
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # print(creds)
    # Refresh or obtain new credentials if needed
    llm = LLMHelper()
    # user = input("Enter the query:")
    response = llm.generate_response(event_input)
    print(f"{response=}")
    summary = response["subject"]
    location = response["location"]
    start_date = response["date"][0]["start_date"]
    end_date = response["date"][0]["end_date"]
    start_time = response["start_time"]
    end_time = response["end_time"]
    listof_dates = response["list_of_dates"]

    def get_timezone():
        # Get the timezone for Kolkata
        timezone = pytz.timezone('Asia/Kolkata')
        return str(timezone)


    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            # print("helloo")
            creds = flow.run_local_server(port=0)
            # print(f"{creds=}")
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
            
    
    try:
        service = build("calendar", "v3", credentials=creds)
        # llm = LLMHelper()
        # user = input("Enter the query:")
        # response = llm.generate_response(user)
        # print(f"{response=}")
        # summary = response["subject"]
        # location = response["location"]
        # start_date = response["date"][0]["start_date"]
        # end_date = response["date"][0]["end_date"]
        # start_time = response["start_time"]
        # end_time = response["end_time"]
        # listof_dates = response["list_of_dates"]
        # print(f"{summary=} | {location=} | {start_date=} | {end_date=} | {start_time=} | {end_time=} | {listof_dates=}")

        end_time = end_time if end_time != "not defined" else "23:59:59"
        end_date = end_date if end_date != "not defined" else start_date
        start_datetime = start_date + "T" + start_time
        end_datetime = end_date + "T" + end_time if end_time != "not defined" else "23:59:59"

        from datetime import datetime

        def minutes_until_event(event_start_time, start_date):
            # Current date and time
            now = datetime.now()
            
            # Event date and time
            event_datetime_str = f"{start_date} {event_start_time}"
            event_datetime = datetime.strptime(event_datetime_str, '%Y-%m-%d %H:%M:%S')
            
            # Calculate the difference
            time_difference = event_datetime - now
            
            # Convert the difference to minutes
            total_minutes = int(time_difference.total_seconds() // 60)
            
            return total_minutes

        totalMin = minutes_until_event(start_time, start_date)
        # Define the event
        event = {
            "summary": summary,
            "location": location,
            "description": "This is description",
            "start": {
                "dateTime": start_datetime,
                "timeZone": get_timezone(),
            },
            "end": {
                "dateTime": end_datetime,
                "timeZone": get_timezone(), 
            },
            "attendees": [
                {"email": "jayantsingh1729@gmail.com"},
                {"email": "singhjk@rknec.edu"},
                {"email": "kurvep18@gmail.com"},
                # {"email": "generaladnan139@gmail.com"},
            ],
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 0},
                    {"method": "email", "minutes": totalMin},
                    {"method": "popup", "minutes": totalMin},
                ],
            },
        }
        
        # Create the event on the calendar
        event_result = service.events().insert(calendarId="primary", body=event).execute()

        events = service.events().list(calendarId="primary", maxResults=10, orderBy="startTime", singleEvents=True)
        # print(dir(events))
        events = events.to_json()
        # print(f"{events=}")
        # for event in events:
        #     print(event)
        
        # Print the event link
        print(f"Event created: {event_result.get('htmlLink')}")
        
    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()