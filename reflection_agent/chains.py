from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI

# ChatPromptTemplate is needed while using chat-based model whereas PromptTemplate was only for plain text input to the model.
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        "You are a hiring manager at a software company in domain of AI/ML shortlisting profiles. Generate critique and recommendations for the user's profile"
        "Always provide detailed recommendations, including requests  for length, virality, style etc..",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        "You are a job seeker assistant tasked with writing excellent profile without hallucination using only information provided by the user."
        "Generate  the best profile details possible for the user's request."
        "If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm



