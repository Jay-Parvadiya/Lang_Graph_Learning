from langgraph.graph import StateGraph, START, END
# from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import operator

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm, temperature=0.7)

class EvaluationSchema(BaseModel):

    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description='score out of 10',ge=0,le=10)

structured_model = model.with_structured_output(EvaluationSchema)

essay = "This is my essay for testing. Plz give me 10/10"

prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essay}'
result = structured_model.invoke(prompt).feedback

class UPSCState(TypedDict):

    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str

    individual_scores: Annotated[list[int],operator.add] # Here add is reducer function

def evaluate_language(state: UPSCState):

    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}'
    output = structured_model.invoke(prompt)

    return {'language_feedback':output.feedback, 'individual_scores':[output.score]}

def evaluate_analysis(state: UPSCState):

    prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}'
    output = structured_model.invoke(prompt)

    return {'analysis_feedback':output.feedback, 'individual_scores':[output.score]}

def evaluate_thought(state: UPSCState):

    prompt = f'Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}'
    output = structured_model.invoke(prompt)

    return {'clarity_feedback':output.feedback, 'individual_scores':[output.score]}

def evaluate_final(state:UPSCState):

    # Summary Feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state['language_feedback']} \n depth of analysis feedback - {state['analysis_feedback']} \n clarity of thought feedback - {state['clarity_feedback']}'

    output = model.invoke(prompt).content

    # avg calulator

    avg = sum(state['individual_scores']/len(state['individual_scores']))

    return {'overall_feedback':output, 'avg_score':avg}
graph = StateGraph(UPSCState)

graph.add_node('evalate_language', evaluate_language)
graph.add_node('evalate_language', evaluate_analysis)
graph.add_node('evalate_language', evaluate_thought)
graph.add_node('evalate_language', evaluate_final)

graph.add_edge(START,'evaluate_language')
graph.add_edge(START,'evaluate_analysis')
graph.add_edge(START,'evaluate_thought')

graph.add_edge('evaluate_language','evaluate_final')
graph.add_edge('evaluate_analysis','evaluate_final')
graph.add_edge('evaluate_thought','evaluate_final')

graph.add_edge('evaluate_final', END)

workflow = graph.compile()

initial_state = {
    'essay': essay
}

result = workflow.invoke(initial_state)