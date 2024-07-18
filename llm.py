from langchain_google_genai import GoogleGenerativeAI
import os
# from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import json

Prompt = """Extract a summary of the sentence provided."""

Elaboration = """
Imagine you have a sentence, like "The cat chased the mouse." Now, suppose you need a shorter version that captures the main idea. That's where the summary comes in. Your task is to tell the language model to take that sentence and condense it into a brief summary. For example, the summary of "The cat chased the mouse" might be "A cat chased a mouse." Keep it simple, just like distilling a long story into a quick headline. Your prompt guides the model to understand and generate concise summaries based on the input sentence.
"""

class Summary:
    def __init__(self):
        pass

    def get_prompt_template(self):
        prompt = PromptTemplate.from_template(Prompt)
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