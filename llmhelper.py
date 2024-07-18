from langchain_google_genai import GoogleGenerativeAI
import os
# from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import json

# load_dotenv(find_dotenv())

prompt_template = """"
Your an intelligent AI model which is designed to perform tasks for finding dates, location and time from sentence and subject of the sentence and donot answer the query asked by user. You should reponse only in JSON format, do not hallucinate, do not give multiple response when asked same question more than once, your answer should be precise and fixed. We are designing to recognize all these thing because we need to schedule some meetings automatically with your help, so the time you find, the dates you should be accurate else it will be a great mistake. The sentence can include dates in various formats such as MM-DD-YYYY or MM-DD-YY or DD-MM-YY or DD-MM-YYYY or YYYY-MM-DD and many of these formats where MM means months and DD means dates and YY means last 2 digits of year and YYYY means year, sometimes dash can replaced with slashes. Date can also be represented in alphanumeric format such as 12 january 2023 or 12 jan 2023, so your tasks is to extract dates from the sentence and store it in a list. Format of date to be stored after extraction is YYYY-MM-DD. Sometimes there can be multiple dates in a sentence, so you should be able to find correctly only two type of dates, namely start date and end date. You are intelligent to recognize which is start date and which is end date. After you successfully found start date and end date, store all the dates in between start date and end date in a list including start date and end date. If there is no start date and end date, then store an empty list. When you calculate dates in between them, you should take care of they are valid dates, do not store invalid date. From invalid I mean, april has 30 days so donot store 31 april in a list.Take care such constraints.Also take care of leap year.Return the output in JSON format only.

Your next job is to extract time from given sentence, so time can be of various formats like 12 hour format or 24 hour format. In 12 hour format you decide time by searching AM or PM in sentence, if it is AM then is morning else evening or night. Another form time can take is like a digit followed by o clock, for example :- 9 o' clock. so here time is 9 AM. After extracting time from sentence your job is to store time in a list in given format. Format for storing time :- HH:MM:SS. User can also provide you such sentences where you are given with multiple time, so you need to recognize which is start time and end time.You have to understand the time very carefully and time can also affect date of meeting so you need to take care of it as well. I will give you an example, Suppose user asked to schedule a meeting on tomorrow from 10 pm to 1 am. You should be very intelligent to recognize that user is asking to store meeting on tomorrow but that will end on day after tomorrow because we are starting meeting tomorrow on night and it is ending on morning of day after tomorrow.Suppose today's date is 19 april 2024. In such cases you should respond :- {{"subject":"This is the subject you recognize from sentence", "start_time":"22:00:00", "end_time":"01:00:00", "date":[{{"start_date":"2024-04-20", "end_date":"2024-04-21"}}]]}}. Time must be in 24 hour format. If the meeting is started at PM and it is ending on AM then it is guaranteed that meeting will end on next day, because meeting is starting at night and it is ending on the morning , so obviously morning will be of next day not same day.
Next job is to extract location from the sentence. This is something you can recognize as a place for meetings, gathering, appointments etc.
You may not find explicit dates or time mentioned in the user query, it can some words that represents time. These words includes today, tomorrow, upcoming, day after tomorrow, and next day, where day can be Sunday, Monday...Saturday. So in this case you will find today's date and calculate asked dates accoring to extracted meaning by you.
You may also find phrases like "from x to y" in sentences. In this case you would have to extract all the dates between x and y including x and y.
After doing all these things, when successfully and accurately calculated start_date and end_date then last job is to find all the dates in between start_date and end_date including start_date and end_date and store them in a list in a given format.
Here is some example to make you understand.


Situation number 1:-
User: Schedule a meet for tomorrow at 3pm at Nagpur.
Here is how you will proceed:- Firstly you will calculate current date, suppose current date is 20 April 2024 if 'tomorrow' is mentioned in it, you will add 1 to the today's date (20 + 1 in this case) and extract date,time and location from it.
AI Response:{{"subject":"Discussion of Project Updates","date": [{{"start_date":"2024-04-21", "end_date":"not defined"}}],"start_time": "15:00:00", "end_time":"not defined","location": "Nagpur","list_of_dates":["2024-04-21"]}}

Situation number 2:-
User: Let's schedule a team meeting for today at 3:00 AM to discuss the project updates at McDonalds.
Here is how you will proceed:- Firstly you will calculate today's date, suppose today's date is 20 April 2024 and extract time and location from it.
AI Response:{{"subject":"Discussion of Project Updates","date": [{{"start_date":"2024-04-20", "end_date":"not defined"}}],"start_time": "03:00:00", "end_time":"not defined","location": "McDonalds Cafe","list_of_dates":["2024-04-20"]}}

Situation number 3:-
User: Book spider-man ticket for 23rd june at 6pm at central park.
AI Response:{{"subject": "Book spider-man ticket","date": [{{"start_date":"2024-06-23", "end_date":"not defined"}}],"start_time": "18:00:00", "end_time":"not defined","location": "Central Park","list_of_dates":["2024-06-23"]}}

Situation number 4:-
User: Remind me to take medicie from 12/02 to 15/02 at 5pm.
You should first find current year and then proceed, suppose current year is 2024.
AI Response:{{"subject": "Medicine","date": [{{"start_date":"2024-02-12", "end_date":"2024-02-13"}}],"2024-02-14","2024-02-15" ,"start_time": "17:00:00", "end_time":"not defined","location": "Not specified","list_of_dates":["2024-02-12", "2024-02-13","2024-02-14","2024-02-15"]}}

Situation number 5:-
User: Schedule a meeting on tomorrow from 10 am to 10pm.
You should first find current date, suppose current date is 20 april 2024 and next you will check whether meeting a ending on next or current day. In this situation meeting is ending on same day so start_date and end_date will be same. Here is how you should respond:-
AI Response: {{"subject":"not specified", "date":[{{"start_date":"2024-04-21", "end_date":"2024-04-21"}}],"start_time":"10:00:00", "end_time":"22:00:00"}},"list_of_dates":["2024-04-21"]]}}

Situation number 6:-
User: Schedule a meeting on tomorrow from 10 pm to 1 am
You should first find current date. Suppose current date is 20 april 2024 and next you will check whether meeting a ending on next or current day. In this situation meeting is ending on next day so start_date and end_will not be same. End date will be 1 day more of start date. Here is how you should respond.
AI Response: {{"subject":"not specified", "date":[{{"start_date":"2024-04-20","end_date":"2024-04-21"}}], "start_time":"22:00:00", "end_time":"01:00:00","list_of_dates":["2024-04-20", "2024-04-21"]}}

Situation number 7:-
User: Remind me to visit gandhinagar tomorrow at 9am.
AI Response:{{"subject": "Visit Gandhinagar","date": [{{"start_date":"2024-04-21", "end_date":"not defined"}}],"start_time": "09:00:00", "end_time":"not defined","location": "Gandhinagar","list_of_dates":["2024-04-21"]}}

User: {query}
AI Response:"""


class LLMHelper:
    def __init__(self):
        pass

    def get_prompt_template(self):
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt
    
    def get_chain(self):
        prompt = self.get_prompt_template()
        llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyAr4lquxKlhwQFTg13iWS7af9wx0H9grMM"
        )
        chain = prompt | llm | StrOutputParser()

        return chain
    
    def generate_response(self,sentence):
        chain = self.get_chain()
        response = chain.invoke({"query": sentence})
        response = json.loads(response)

        return response
    
    

# chain = LLMHelper.get_chain()

# def get_chai
# # query = input("Enter text: ")
# query = "Schedule a meeting on 23rd june to 26th june from 10am at radison blue for development of rcoem."
# response = chain.invoke({"query": query})
# response = json.loads(response)
# print(response)
