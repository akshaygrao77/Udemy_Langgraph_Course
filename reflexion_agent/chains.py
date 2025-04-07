import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,PydanticToolsParser)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate

from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion,ReviseAnswer


llm = ChatOpenAI(model="gpt-4o-mini")
# Return function call from LLM and transform to dictionary
parser = JsonOutputToolsParser(return_id=True)
# Take response from LLM and search for function call and return in this format
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are expert researcher.
        Current time:{time}
        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. Recommend search queries to research information and improve your answer.
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system","Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(first_instruction="Provide a detailed ~250 word answer.")

# Here bind tool idea is used over the pydantic object just to ground the LLMs answer to whatever we want to with pydantic schema
first_responder = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")

revise_instructions = """ Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You must include numerical citations  in your revised answer to ensure it can be verified.
    - Add a "Reference" section to the bottom of your answer (which does not count towards the word limit). In form of:
        - [1] https://example.com
        - [2] https://example2.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

revisor = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(tools=[ReviseAnswer],tool_choice="ReviseAnswer")

if __name__=='__main__':
    human_message = HumanMessage(content="Write about AI powered SOC/autonomous SOC domain, list the startups that do that and raised capital.")

    chain = (first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion") | parser_pydantic)

    # messages key since the chatprompttemplate is designed that way
    res = chain.invoke(input={"messages":[human_message]})

    print(res)
