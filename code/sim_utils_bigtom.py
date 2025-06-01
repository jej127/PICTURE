# Author: Shawn Lee
# Date: Aug 2023
# Description: All utilities for running simulated ToM on BigToM.

from llm_utils import *
from prompts_bigtom import *

def percept_2_story(perception, coi):
    p = eval(perception)
    story = ''
    for d in p:
        for k,v in d.items():
            if coi in v:
                story += (' ' + k)
    return story.strip()

class Agent:
    """Agent class for simulation.
    """
    def __init__(self, llm:LLM, name=""):
        self.name = name
        self.perspective = ""
        self.llm = llm
        self.debug = False
        if "llama" in self.llm.model_name:
            # Llama evaluation prompt.
            self.evalPrompt = LLAMA_SIM_PROMPT
        else:
            # ChatGPT evaluation prompt.
            self.evalPrompt = GPT_SIM_PROMPT
    
        # Debugging
        self.wasAsked = None
        self.replied = None

    
    def evalQuestion(self, question:str) -> str:
        """Simulation function.

        Args:
            question (str): BigToM question + answer choices

        Returns:
            str: Answer choice
        """
        
        prompt = self.evalPrompt.format(perspective=self.perspective, name=self.name, question=question)
        choice = self.llm.getOutput(prompt)
        self.wasAsked, self.replied = prompt, choice
        return choice

class Agent2:
    """Agent class for simulation.
    """
    def __init__(self, llm:LLM, name=""):
        self.name = name
        self.perspective = ""
        self.story = ""
        self.llm = llm
        self.debug = False
        if "llama" in self.llm.model_name:
            # Llama evaluation prompt.
            self.evalPrompt = LLAMA_SIM_PROMPT2
        else:
            # ChatGPT evaluation prompt.
            self.evalPrompt = GPT_SIM_PROMPT2
    
        # Debugging
        self.wasAsked = None
        self.replied = None

    
    def evalQuestion(self, question:str) -> str:
        """Simulation function.

        Args:
            question (str): BigToM question + answer choices

        Returns:
            str: Answer choice
        """
        
        prompt = self.evalPrompt.format(story=self.story, perspective=self.perspective, name=self.name, question=question)
        choice = self.llm.getOutput(prompt)
        self.wasAsked, self.replied = prompt, choice
        return choice

class World:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, agent:Agent = None, perspective:str = None, knowsChange:bool = None, simModel=None, ptom=False):
        self.llm = llm
        self.agent = agent     
        self.context = context # Original BigToM Question
        self.perspective = perspective # Perspective
        self.knowsChange = knowsChange # Does the agent know the change?
        self.agentName = None 
        self.simModel = simModel
        self.ptom = ptom
        if self.simModel == None: # If sim model not specified, we are just using the same model for PT and simulation.
            self.simModel = self.llm
        
    def takePerspective(self) -> None:
        """Function that takes perspective.
        """
        
        self.agentName = self.context.split(" ")[0].replace(',','')
        
        story = self.context
        name = self.agentName
        
        if self.ptom:
            # Prompt for GPT
            prompt = GPT_PERSPECTIVE_PROMPT_PT
            if "llama" in self.simModel.model_name:
                # Prompt for llama, as it outputs useless things
                prompt = LLAMA_PERSPECTIVE_PROMPT_PT

            perspective = self.simModel.getOutput(prompt.format(story=story)).strip()
            try:
                eval(perspective)
            except:
                if "llama" in self.simModel.model_name:
                    perspective = ''.join(['['] + perspective.split('[')[1:])
                    perspective = perspective.replace(':',': [')
                else:
                    perspective = perspective.split('\n[')[1]
                    perspective = perspective.split(']\n')[0]
                    perspective = ''.join(['[',perspective,']'])

            try:
                perspective = percept_2_story(perspective, name)
            except:
                perspective = 'Invalid Request Error Occurred'

        else:
            # Prompt for GPT
            prompt = GPT_PERSPECTIVE_PROMPT
            if "llama" in self.simModel.model_name:
                # Prompt for llama, as it outputs useless things
                prompt = LLAMA_PERSPECTIVE_PROMPT

            perspective = self.simModel.getOutput(prompt.format(story=story, name=name)).split(":")[-1].strip()

        if self.perspective is None: # using model, not perspective gold
            self.perspective = perspective


    def setupAgent(self) -> None:
        # Create agent if no agent was given (which is always lol. just futureproofing)
        if self.agent is None:
            self.agent = Agent(self.simModel, "")
        
        # Give it its name! (play god a little here)        
        self.agent.name = self.agentName
        
        # Give it the perspective
        self.agent.perspective = self.perspective

    # Debugging Function
    def getWorldState(self) -> str:
        return f"""\
Agent Name: {self.agentName}

Context: {self.context}

Perspective: {self.perspective}
""" 
    
    # Debugging Function
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:
{self.agent.wasAsked}

and outputted:
{self.agent.replied}
""")

    

class World_ours:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, agent:Agent = None, perspective:str = None, knowsChange:bool = None, simModel=None, question=None):
        self.llm = llm
        self.agent = agent     
        self.context = context # Original BigToM Question
        self.perspective = perspective # Perspective
        self.knowsChange = knowsChange # Does the agent know the change?
        self.agentName = None 
        self.simModel = simModel
        self.question = question
        if self.simModel == None: # If sim model not specified, we are just using the same model for PT and simulation.
            self.simModel = self.llm
        
    def takePerspective(self) -> None:
        """Function that takes perspective.
        """
        
        self.agentName = self.context.split(" ")[0].replace(',','')
        
        story = self.context
        name = self.agentName
        
        # Prompt for GPT
        prompt = GPT_PERSPECTIVE_PROMPT_OURS
        if "llama" in self.simModel.model_name:
            # Prompt for llama, as it outputs useless things
            prompt = LLAMA_PERSPECTIVE_PROMPT_OURS

        perspective = self.simModel.getOutput(prompt.format(story=story, name=name, question=self.question)).strip()

        if self.perspective is None: # using model, not perspective gold
            self.perspective = perspective


    def setupAgent(self) -> None:
        # Create agent if no agent was given (which is always lol. just futureproofing)
        if self.agent is None:
            self.agent = Agent(self.simModel, "")
        
        # Give it its name! (play god a little here)        
        self.agent.name = self.agentName
        
        # Give it the perspective
        self.agent.perspective = self.perspective

    # Debugging Function
    def getWorldState(self) -> str:
        return f"""\
Agent Name: {self.agentName}

Context: {self.context}

Perspective: {self.perspective}
""" 

    # Debugging Function
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:
{self.agent.wasAsked}

and outputted:
{self.agent.replied}
""")


class World_ours2:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, agent:Agent = None, perspective:str = None, knowsChange:bool = None, simModel=None, question=None):
        self.llm = llm
        self.agent = agent     
        self.context = context # Original BigToM Question
        self.perspective = perspective # Perspective
        self.story = context
        self.knowsChange = knowsChange # Does the agent know the change?
        self.agentName = None 
        self.simModel = simModel
        self.question = question
        if self.simModel == None: # If sim model not specified, we are just using the same model for PT and simulation.
            self.simModel = self.llm
        
    def takePerspective(self) -> None:
        """Function that takes perspective.
        """
        
        self.agentName = self.context.split(" ")[0].replace(',','')
        
        story = self.context
        name = self.agentName
        
        # Prompt for GPT
        prompt = GPT_PERSPECTIVE_PROMPT_PS
        if "llama" in self.simModel.model_name:
            # Prompt for llama, as it outputs useless things
            prompt = LLAMA_PERSPECTIVE_PROMPT_PS

        perspective = self.simModel.getOutput(prompt.format(story=story, name=name, question=self.question)).strip()

        if self.perspective is None: # using model, not perspective gold
            self.perspective = perspective


    def setupAgent(self) -> None:
        # Create agent if no agent was given (which is always lol. just futureproofing)
        if self.agent is None:
            self.agent = Agent2(self.simModel, "")
        
        # Give it its name! (play god a little here)        
        self.agent.name = self.agentName
        
        # Give it the perspective
        self.agent.perspective = self.perspective
        self.agent.story = self.story

    # Debugging Function
    def getWorldState(self) -> str:
        return f"""\
Agent Name: {self.agentName}

Context: {self.context}

Perspective: {self.perspective}
""" 

    # Debugging Function
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:
{self.agent.wasAsked}

and outputted:
{self.agent.replied}
""")


# Evaluation function
def evalQuestion(llm:ChatGPT, context:str, question:str, knowsChange:bool, perspectiveGold:bool, simModel=None, ptom=False) -> Tuple[str, World]:
    if perspectiveGold: # give gold labels for perspective, knowsChange by parsing sentence and question type
        perspective = '.'.join(context.split('.')[:-3]) + '.'
        knowsChange = knowsChange
    else:
        perspective = None
        knowsChange = None
    world = World(llm, context, perspective=perspective, knowsChange=knowsChange, simModel=simModel, ptom=ptom)
    world.takePerspective()
    world.setupAgent()
    answer = world.agent.evalQuestion(question)
    return answer, world


# Evaluation function
def evalQuestion_ours(llm:ChatGPT, context:str, question:str, knowsChange:bool, perspectiveGold:bool, simModel=None) -> Tuple[str, World]:
    if perspectiveGold: # give gold labels for perspective, knowsChange by parsing sentence and question type
        perspective = '.'.join(context.split('.')[:-3]) + '.'
        knowsChange = knowsChange
    else:
        perspective = None
        knowsChange = None
    world = World_ours(llm, context, perspective=perspective, knowsChange=knowsChange, simModel=simModel, question=question)
    world.takePerspective()
    world.setupAgent()
    answer = world.agent.evalQuestion(question)
    return answer, world


# Evaluation function
def evalQuestion_ours2(llm:ChatGPT, context:str, question:str, knowsChange:bool, perspectiveGold:bool, simModel=None) -> Tuple[str, World]:
    if perspectiveGold: # give gold labels for perspective, knowsChange by parsing sentence and question type
        perspective = '.'.join(context.split('.')[:-3]) + '.'
        knowsChange = knowsChange
    else:
        perspective = None
        knowsChange = None
    world = World_ours2(llm, context, perspective=perspective, knowsChange=knowsChange, simModel=simModel, question=question)
    world.takePerspective()
    world.setupAgent()
    answer = world.agent.evalQuestion(question)
    return answer, world


# Function for ablation
def oneBigPrompt(llm, story, question):
    name = story.split()[0]
    prompt = ABLATION_ONEPROMPT_SIM
    output = llm.getOutput(prompt.format(name=name, story=story, question=question))
    answer = output[output.rfind("Answer:")+len("Answer:"):].strip()
    return answer, output
