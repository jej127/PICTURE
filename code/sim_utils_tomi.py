# Author: Shawn Lee
# Date: Aug 2023
# Description: All utilities for running simulated ToM on ToMi.

from llm_utils import *
from prompts_tomi import *

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
    def __init__(self, llm, name=""):
        self.name = name
        self.perspective = ""
        self.llm = llm
        
        # Here, we differ the simulation prompts per model.
        if "llama" in llm.model_name:
            # Simulation prompt for Llama
            self.evalPrompt = LLAMA_SIM_PROMPT
        else: 
            # Simulation prompt for ChatGPT
            self.evalPrompt = GPT_SIM_PROMPT
        # DEBUGGING PURPOSES
        self.wasAsked = None
        self.replied = None
        self.debug = False
    
    def evalQuestion(self, question:str) -> str:
        """Answers BigToM question based on agent belief.

        Args:
            question (str): BigToM question + answer choices

        Returns:
            str: Answer choice
        """
        prompt = self.evalPrompt.format(perspective=self.perspective, name=self.name, question=question)
        choice = self.llm.getOutput(prompt)
        self.wasAsked, self.replied = prompt, choice
        return choice, self.perspective # Return perspective for debugging purposes

class Agent2:
    """Agent class for simulation.
    """
    def __init__(self, llm, name=""):
        self.name = name
        self.story = ""
        self.perspective = ""
        self.llm = llm
        
        # Here, we differ the simulation prompts per model.
        if "llama" in llm.model_name:
            # Simulation prompt for Llama
            self.evalPrompt = LLAMA_SIM_PROMPT2
        else: 
            # Simulation prompt for ChatGPT
            self.evalPrompt = GPT_SIM_PROMPT2
        # DEBUGGING PURPOSES
        self.wasAsked = None
        self.replied = None
        self.debug = False
    
    def evalQuestion(self, question:str) -> str:
        """Answers BigToM question based on agent belief.

        Args:
            question (str): BigToM question + answer choices

        Returns:
            str: Answer choice
        """
        prompt = self.evalPrompt.format(story=self.story, perspective=self.perspective, name=self.name, question=question)
        choice = self.llm.getOutput(prompt)
        self.wasAsked, self.replied = prompt, choice
        return choice, self.perspective # Return perspective for debugging purposes

class World:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, debug=False, simModel=None, ptom=False):
        self.llm = llm
        self.agentNames = []
        self.agents:Dict[str, Agent] = {}                 # Note this is a dictioary for maybe future multi-agent simulations
        self.context = context                            # Original ToMi Question (JSON format)
        self.story:str = context["story"]                 # ToMI story
        self.perspectives : Dict[str, str] = {}           # Perspectives (Dict for same reason as above)
        self.debug = debug                                # This is verbose
        self.simModel = simModel
        self.ptom = ptom
        if self.simModel == None:
            self.simModel = self.llm
        
    def parseCharacters(self) -> None:
        # We use ChatGPT here a little bit, just to parse the characters in this story.
        prompt = f"""\
{self.story}
What are the characters in this story?
Output only the character names, separated by commas."""
        gpt = ChatGPT("gpt-3.5-turbo")
        self.agentNames = gpt.getOutput(prompt).replace(" ", "").split(",")
        if self.debug:
            print("Agent names:", self.agentNames)
    
    def takePerspective(self, characterName=None) -> None:
        # Here is the perspective-taking.
        if self.ptom:
            prompt = PERSPECTIVE_PROMPT_PT
        else:
            prompt = PERSPECTIVE_PROMPT
            
        if characterName is None:
        # Then parse perspective for each character. We don't use this though.
            for character in self.agentNames:
                self.perspectives[character] = self.llm.getOutput(prompt.format(story=self.story, character=character))
                if self.debug:
                    print(f"Perspective of {character}:", self.perspectives[character])
        else:
            # Take perspective for given character.
            # Take perspective for given character.
            result = self.llm.getOutput(prompt.format(story=self.story, character=characterName))
            if self.ptom:
                try:
                    eval(result)
                except:
                    if "llama" in self.simModel.model_name:
                        result = ''.join(['['] + result.split('[')[1:])
                        result = result.replace(':',': [')
                    else:
                        result = result.split('\n[')[1]
                        result = result.split(']\n')[0]
                        result = ''.join(['[',result,']'])

                try:
                    result = percept_2_story(result, characterName)
                except:
                    result = 'Invalid Request Error Occurred'

            self.perspectives[characterName] = result
            if self.debug:
                print(f"Perspective of {characterName}:", self.perspectives[characterName])
        

    def setupAgent(self) -> None:
        """Create agents.
        """
        for agentName, perspective in self.perspectives.items():
            agent = Agent(self.simModel, name=agentName)
            agent.perspective = perspective
            # Add to agent dictionary
            self.agents[agentName] = agent
        
        
    def getWorldState(self) -> str:
        return f"""\
TBD! I don't have anything here yet.
""" 
    
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:

{self.agent.wasAsked}

and outputted:

{self.agent.replied}
""")

    def evalQuestion(self, question, agentName=None) -> str:
        if agentName is None:
            # This is a question about the truth.
            # For truth questions, there's really no point of simulation/perspective taking, so we just ask the LLM the question (same as baseline).
            return self.llm.getOutput(f"{self.story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        else:
            # Here we ask the agent to simulate.
            return self.agents[agentName].evalQuestion(question)
        

class World_ours:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, debug=False, simModel=None, question=None):
        self.llm = llm
        self.agentNames = []
        self.agents:Dict[str, Agent] = {}                 # Note this is a dictioary for maybe future multi-agent simulations
        self.context = context                            # Original ToMi Question (JSON format)
        self.story:str = context["story"]                 # ToMI story
        self.perspectives : Dict[str, str] = {}           # Perspectives (Dict for same reason as above)
        self.debug = debug                                # This is verbose
        self.simModel = simModel
        self.question = question
        if self.simModel == None:
            self.simModel = self.llm
        
    def parseCharacters(self) -> None:
        # We use ChatGPT here a little bit, just to parse the characters in this story.
        prompt = f"""\
{self.story}
What are the characters in this story?
Output only the character names, separated by commas."""
        gpt = ChatGPT("gpt-3.5-turbo")
        self.agentNames = gpt.getOutput(prompt).replace(" ", "").split(",")
        if self.debug:
            print("Agent names:", self.agentNames)
    
    def takePerspective(self, characterName=None) -> None:
        
        # Here is the perspective-taking.
        prompt = PERSPECTIVE_PROMPT_OURS
        if "llama" in self.simModel.model_name:
            # Prompt for llama, as it outputs useless things
            prompt = PERSPECTIVE_PROMPT_OURS_LLAMA
        
        if characterName is None:
        # Then parse perspective for each character. We don't use this though.
            for character in self.agentNames:
                self.perspectives[character] = self.llm.getOutput(prompt.format(story=self.story, question=self.question, character=character))
                if self.debug:
                    print(f"Perspective of {character}:", self.perspectives[character])
        else:
            # Take perspective for given character.
            


            self.perspectives[characterName] = self.llm.getOutput(prompt.format(story=self.story, question=self.question, character=characterName))
            if self.debug:
                print(f"Perspective of {characterName}:", self.perspectives[characterName])
        

    def setupAgent(self) -> None:
        """Create agents.
        """
        for agentName, perspective in self.perspectives.items():
            agent = Agent(self.simModel, name=agentName)
            agent.perspective = perspective
            # Add to agent dictionary
            self.agents[agentName] = agent
        
        
    def getWorldState(self) -> str:
        return f"""\
TBD! I don't have anything here yet.
""" 
    
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:

{self.agent.wasAsked}

and outputted:

{self.agent.replied}
""")

    def evalQuestion(self, question, agentName=None) -> str:
        if agentName is None:
            # This is a question about the truth.
            # For truth questions, there's really no point of simulation/perspective taking, so we just ask the LLM the question (same as baseline).
            return self.llm.getOutput(f"{self.story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        else:
            # Here we ask the agent to simulate.
            return self.agents[agentName].evalQuestion(question)



class World_ours2:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, debug=False, simModel=None, question=None):
        self.llm = llm
        self.agentNames = []
        self.agents:Dict[str, Agent] = {}                 # Note this is a dictioary for maybe future multi-agent simulations
        self.context = context                            # Original ToMi Question (JSON format)
        self.story:str = context["story"]                 # ToMI story
        self.perspectives : Dict[str, str] = {}           # Perspectives (Dict for same reason as above)
        self.debug = debug                                # This is verbose
        self.simModel = simModel
        self.question = question
        if self.simModel == None:
            self.simModel = self.llm
        
    def parseCharacters(self) -> None:
        # We use ChatGPT here a little bit, just to parse the characters in this story.
        prompt = f"""\
{self.story}
What are the characters in this story?
Output only the character names, separated by commas."""
        gpt = ChatGPT("gpt-3.5-turbo")
        self.agentNames = gpt.getOutput(prompt).replace(" ", "").split(",")
        if self.debug:
            print("Agent names:", self.agentNames)
    
    def takePerspective(self, characterName=None) -> None:
        
        # Here is the perspective-taking.
        prompt = PERSPECTIVE_PROMPT_PS
        if "llama" in self.simModel.model_name:
            # Prompt for llama, as it outputs useless things
            prompt = PERSPECTIVE_PROMPT_PS
        
        if characterName is None:
        # Then parse perspective for each character. We don't use this though.
            for character in self.agentNames:
                self.perspectives[character] = self.llm.getOutput(prompt.format(story=self.story, question=self.question, character=character))
                if self.debug:
                    print(f"Perspective of {character}:", self.perspectives[character])
        else:
            # Take perspective for given character.
            


            self.perspectives[characterName] = self.llm.getOutput(prompt.format(story=self.story, question=self.question, character=characterName))
            if self.debug:
                print(f"Perspective of {characterName}:", self.perspectives[characterName])
        

    def setupAgent(self) -> None:
        """Create agents.
        """
        for agentName, perspective in self.perspectives.items():
            agent = Agent2(self.simModel, name=agentName)
            agent.perspective = perspective
            agent.story = self.story
            # Add to agent dictionary
            self.agents[agentName] = agent
        
        
    def getWorldState(self) -> str:
        return f"""\
TBD! I don't have anything here yet.
""" 
    
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:

{self.agent.wasAsked}

and outputted:

{self.agent.replied}
""")

    def evalQuestion(self, question, agentName=None) -> str:
        if agentName is None:
            # This is a question about the truth.
            # For truth questions, there's really no point of simulation/perspective taking, so we just ask the LLM the question (same as baseline).
            return self.llm.getOutput(f"{self.story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        else:
            # Here we ask the agent to simulate.
            return self.agents[agentName].evalQuestion(question)


class World_pre:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:str, cause_d:list, p_type=0, debug=False, simModel=None, ptom=False):
        self.llm = llm
        self.agentNames = []
        self.agents:Dict[str, Agent] = {}                 # Note this is a dictioary for maybe future multi-agent simulations
        self.context = context                            # Original ToMi Question (JSON format)
        self.cause_d = cause_d
        self.p_type = p_type
        self.story:str = context["story"]                 # ToMI story
        self.perspectives : Dict[str, str] = {}           # Perspectives (Dict for same reason as above)
        self.debug = debug                                # This is verbose
        self.simModel = simModel
        self.ptom = ptom
        if self.simModel == None:
            self.simModel = self.llm
        
    def parseCharacters(self) -> None:
        # We use ChatGPT here a little bit, just to parse the characters in this story.
        prompt = f"""\
{self.story}
What are the characters in this story?
Output only the character names, separated by commas."""
        gpt = ChatGPT("gpt-3.5-turbo")
        self.agentNames = gpt.getOutput(prompt).replace(" ", "").split(",")
        if self.debug:
            print("Agent names:", self.agentNames)
    
    def takePerspective(self, characterName=None) -> None:
        events = self.story.strip().split('\n')
        #question = self.context['question']
        #target_object = question.replace('?','').split()[-1]
        #relocate_event = [e for e in events if 'moved' in e][0]
        temp = self.story.split(self.context['answer'] + ' is in the ')[1]
        main_location = temp.split('.')[0]

        start_perception_list = [e for e in events if all([characterName in e, main_location in e, ('entered' in e or 'is in' in e)])]
        #assert len(start_perception_list) == 1, start_perception_list
        start_perception = start_perception_list[0]
        start_idx = int(start_perception.split()[0])

        end_perception_list = [e for e in events if all([characterName in e, main_location in e, 'exited' in e])]
        assert len(end_perception_list) <= 1, end_perception_list
        if len(end_perception_list) == 1:
            if len(start_perception_list) > 1:
                end_idx = len(events)
            else:
                end_perception = end_perception_list[0]
                end_idx = int(end_perception.split()[0])
        else: 
            end_idx = len(events)

        assert start_idx < end_idx, events

        perspective = []

        if self.p_type >= 0:
            for j in range(start_idx-1, end_idx):
                e, cause_label = self.cause_d[j]
                assert e == events[j]
                if (cause_label or self.p_type in [0,2]):
                    perspective.append(e)

            if self.p_type>=2:
                for j in range(end_idx, len(events)):
                    e, cause_label = self.cause_d[j]
                    assert e == events[j]
                    if cause_label and 'is in' not in e:
                        e = e.split()[:1] + [f'{characterName} is not aware of the events happening in {main_location} anymore.']
                        e = " ".join(e)
                        perspective.append(e.replace(' The ',' the '))
        else:
            for j in range(len(events)):
                e, cause_label = self.cause_d[j]
                assert e == events[j]
                perspective.append(e)

        perspective_ = []
        for j,e in enumerate(perspective):
            e_ = " ".join([str(j+1)] + e.split()[1:])
            perspective_.append(e_)

        self.perspectives[characterName] = '\n'.join(perspective_)
        if self.debug:
            print(f"Perspective of {characterName}:", self.perspectives[characterName])

    def setupAgent(self) -> None:
        """Create agents.
        """
        for agentName, perspective in self.perspectives.items():
            agent = Agent(self.simModel, name=agentName)
            agent.perspective = perspective
            # Add to agent dictionary
            self.agents[agentName] = agent
        
        
    def getWorldState(self) -> str:
        return f"""\
TBD! I don't have anything here yet.
""" 
    
    def getAgentState(self) -> str:
        return(f"""
Agent was prompted:

{self.agent.wasAsked}

and outputted:

{self.agent.replied}
""")

    def evalQuestion(self, question, agentName=None) -> str:
        if agentName is None:
            # This is a question about the truth.
            # For truth questions, there's really no point of simulation/perspective taking, so we just ask the LLM the question (same as baseline).
            return self.llm.getOutput(f"{self.story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        else:
            # Here we ask the agent to simulate.
            return self.agents[agentName].evalQuestion(question)


# Evaluation function
def evalQuestion(llm:ChatGPT, context:str, question:str, debug=False, simModel=None, ptom=False) -> Tuple[str, str]:
    """End to end function for Tomi evaluation.
    """
    # Create world for perspective taking
    world = World(llm, context, debug=debug, simModel=simModel, ptom=ptom)
    
    # What's the subject of the question (who's perspective do we have to take?)
    questionSubject = question.split(" ")[2]
    
    # What are the characters in this story?
    world.parseCharacters()
    
    # Is this a reality/memory question?
    if questionSubject not in world.agentNames:
        # Question about zeroth-order belief (Reality/Memory question)
        answer, perspective = world.evalQuestion(question)
        return answer, perspective
    
    # Else, run the pipeline.
    
    # 1. Perspective-taking
    world.takePerspective(characterName=questionSubject)
    world.setupAgent()
    
    # 2. Simulation
    answer, perspective = world.evalQuestion(question, agentName=questionSubject)
    
    # ...and we have our answer!
    return answer, perspective


def evalQuestion_ours(llm:ChatGPT, context:str, question:str, debug=False, simModel=None) -> Tuple[str, str]:
    """End to end function for Tomi evaluation.
    """
    # Create world for perspective taking
    world = World_ours(llm, context, debug=debug, simModel=simModel, question=question)
    
    # What's the subject of the question (who's perspective do we have to take?)
    questionSubject = question.split(" ")[2]
    
    # What are the characters in this story?
    world.parseCharacters()
    
    # Is this a reality/memory question?
    if questionSubject not in world.agentNames:
        #questionSubject = 'the omniscent viewer'
        # Question about zeroth-order belief (Reality/Memory question)
        answer, perspective = world.evalQuestion(question)
        return answer, perspective

    # Else, run the pipeline.
    
    # 1. Perspective-taking
    world.takePerspective(characterName=questionSubject)
    world.setupAgent()
    
    # 2. Simulation
    answer, perspective = world.evalQuestion(question, agentName=questionSubject)
    
    # ...and we have our answer!
    return answer, perspective


def evalQuestion_pre(llm:ChatGPT, context:str, question:str, cause_d:list, p_type=0, debug=False, simModel=None, ptom=False) -> Tuple[str, str]:
    """End to end function for Tomi evaluation.
    """
    # Create world for perspective taking
    world = World_pre(llm, context, cause_d, p_type, debug=debug, simModel=simModel, ptom=ptom)
    
    # What's the subject of the question (who's perspective do we have to take?)
    questionSubject = question.split(" ")[2]
    
    # What are the characters in this story?
    world.parseCharacters()
    
    # Is this a reality/memory question?
    if questionSubject not in world.agentNames:
        # Question about zeroth-order belief (Reality/Memory question)
        answer, perspective = world.evalQuestion(question)
        return answer, perspective
    
    # Else, run the pipeline.
    
    # 1. Perspective-taking
    world.takePerspective(characterName=questionSubject)
    world.setupAgent()
    
    # 2. Simulation
    answer, perspective = world.evalQuestion(question, agentName=questionSubject)

    #answer, perspective = 'abc', world.perspectives[questionSubject]
    
    # ...and we have our answer!
    return answer, perspective


def evalQuestion_ours2(llm:ChatGPT, context:str, question:str, debug=False, simModel=None) -> Tuple[str, str]:
    """End to end function for Tomi evaluation.
    """
    # Create world for perspective taking
    world = World_ours2(llm, context, debug=debug, simModel=simModel, question=question)
    
    # What's the subject of the question (who's perspective do we have to take?)
    questionSubject = question.split(" ")[2]
    
    # What are the characters in this story?
    world.parseCharacters()
    
    # Is this a reality/memory question?
    if questionSubject not in world.agentNames:
        #questionSubject = 'the omniscent viewer'
        # Question about zeroth-order belief (Reality/Memory question)
        answer, perspective = world.evalQuestion(question)
        return answer, perspective

    # Else, run the pipeline.
    
    # 1. Perspective-taking
    world.takePerspective(characterName=questionSubject)
    world.setupAgent()
    
    # 2. Simulation
    answer, perspective = world.evalQuestion(question, agentName=questionSubject)
    
    # ...and we have our answer!
    return answer, perspective



def oneBigPrompt(llm, story, question):
    """
    Function for the one-prompt-simulation ablation.
    """
    
    # What's the subject of the question (who's perspective do we have to take?)
    character = question.split(" ")[2]
    prompt = f"""\
{story}
What are the characters in this story?
Output only the character names, separated by commas."""
    agentNames = llm.getOutput(prompt).replace(" ", "").split(",")
    # Is this a reality/memory question?
    if character not in agentNames:
        answer, perspective = llm.getOutput(f"{story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        return answer, perspective
    # Else, run the one-prompt simulation.
    prompt = ABLATION_ONEPROMPT_SIM
    output = llm.getOutput(prompt.format(character=character, story=story, question=question))
    # Parse output.
    # ChatGPT (unlike llama-2) is pretty good at following formatting instructions, so this pretty much never fails. 
    if "Answer:" in output:
        answer = output[output.rfind("Answer:")+len("Answer:"):].strip()
    else:
        answer = output.split('\n')[-1].strip()
    return answer, output
