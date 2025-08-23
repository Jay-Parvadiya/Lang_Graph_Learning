from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):

    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    boundry: float

    summary: str

def calculate_sr(state: BatsmanState):

    sr = (state['runs']/state['balls'])*100
    return {'sr':sr}

def calculate_bpb(state: BatsmanState):

    bpb = state['balls']/(state['fours'] + state['sixes'])
    return {'bpb':bpb}

def calculate_boundary(state: BatsmanState) -> BatsmanState:

    boundary = ((state['fours']*4) + (state['sixes']*6)/state['runs'])*100
    return {'boundry':boundary}

def summary(state: BatsmanState) -> BatsmanState:

    summary = f"""
    Strike Rate - {state['sr']} \n
    Balls per boundary - {state['bpb']} \n
    Boundary percenet - {state['boundry']}
    """
    state['summary'] = summary
    return state


graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary', calculate_boundary)
graph.add_node('summary', summary)

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary', 'summary')

graph.add_edge('summary',END)

workflow = graph.compile()

inital_state = {
    'runs':100,
    'balls':50,
    'fours':6,
    'sixes':4,
}

final_result = workflow.invoke(inital_state)

print(final_result)