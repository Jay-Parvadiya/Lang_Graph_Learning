from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm,temperature=0.7)

class SentimentSchema(BaseModel):

    sentiment: Literal['positive','negative'] = Field(description='Sentiment of the review')


class ReviewState(TypedDict):

    review : str
    sentiment: Literal['positive','nagetive']
    diagnosis: dict
    response: str

structured_model = model.with_structured_output(SentimentSchema)

prompt = 'What is the sentiment of the following review - The softwer is too bad'

def find_sentiment(state: ReviewState):
    prompt = f'For the follwing review find out the sentiment \n {state["review"]}'
    sentiment = structured_model.invoke(prompt).sentiment

    return {'sentiment':sentiment}

def check_sentiment(state: ReviewState) -> Literal['positive_response','run_diagnosis']:

    if state['sentiment'] == 'positive':
        return 'positive_response'
    else :
        return 'run_diagnosis'

def positive_response(state: ReviewState):
    prompt = f'Write a warm thank-you message in response to this review: \n\n\"{state['review']}\"\n Also, kindly ask the use to leave feedback on our website'

    response = model.invoke(prompt).content

    return {'response':response}

def run_diagnosis(state: ReviewState):
    prompt = f'Diagnose this nagative reveiw: \n\n{state['review']}\n Return issue_type, tone and urgency'
    response = model.invoke(prompt)


    return {'diagonsis':response.model_dump()}

def nagative_response(state: ReviewState):
    diagnosis = state['diagnosis']
    prompt = 'Write an empathetic response for nagative review'
    response = model.invoke(prompt).content

    return {'response':response}
graph = StateGraph(ReviewState)

graph.add_node('find_sentiment',find_sentiment)
graph.add_node('positive_response',positive_response)
graph.add_node('run_diagnosis',run_diagnosis)
graph.add_node('nagative_response', nagative_response)

graph.add_edge(START,'find_sentiment')
graph.add_conditional_edges('find_sentiment', check_sentiment)
graph.add_edge('positive_response',END)
graph.add_edge('run_diagnosis','nagative_response')
graph.add_edge('nagative_response',END)


workflow = graph.compile()

intial_state={
    'review':'The product is bad'
}


output = workflow.invoke(intial_state)
