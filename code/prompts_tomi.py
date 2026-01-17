
PERSPECTIVE_PROMPT = """\
The following is a sequence of events about some characters, that takes place in multiple locations.
Your job is to output only the events that the specified character, {character}, knows about.
Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

Story:
{story}

What events does {character} know about? Only output the events according to the above rules, do not provide an explanation."""


PERSPECTIVE_PROMPT_OURS = """\
The following is a sequence of events about some characters, that takes place in multiple locations.
Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

Story:
{story}

The main question to be solved is as follows:

{question}

What events does and does not {character} know about? Output the answer for the main question by thinking step by step."""

PERSPECTIVE_PROMPT_OURS_LLAMA = """\
The following is a sequence of events about some characters, that takes place in multiple locations.
Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

Story:
{story}

The main question to be solved is as follows:

{question}
This is the end of the main question.

What events does and does not {character} know about? Output the answer for the main question by thinking step by step."""

PERSPECTIVE_PROMPT_PT="""\
Story:
{story}

Create a JSON array consisting of JSON objects. Each object should contain a sentence from the story and the perceivers of the scene described in that sentence. Assume that characters in the story can perceive every scene occurring in their location but not scenes occurring elsewhere. Also, include the actant of any action as a perceiver of that action.
Provide only a JSON array in the following format. Do not include any explanation.
[{{"Noah exited the living room.": ["Noah", "Emma"]}},]

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location."""

PERSPECTIVE_PROMPT_PS="""\
Story:
{story}

Question:
{question}

Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
"""

LLAMA_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
You must choose one of the above choices, do not say there is not enough information. Answer with a single word, do not output anything else. 
"""

GPT_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
Keep your answer concise, one sentence is enough. You must choose one of the above choices.
"""


######## Prompts for other methods ##########

baselinePrompt = """\
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
"""
       
questionPrompt = """\
{question}
Choose from the following:
{containers_0}, {containers_1}
""" 

questionPromptCoT = """\
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.
""" 


rulesPrompt = """\
I will give you a story and ask you a question. Answer the question based on the rules.

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Keep your answer concise. Answer with a single word.
"""

rulesCoTPrompt = """\
I will give you a story and ask you a question. Answer the question based on the rules.

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.
"""

cotPrompt = """
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.
"""

rulespsPrompt = """
I will give you a story and ask you a question. Answer the question based on the rules.

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step. Write your answer in the end as 'Answer: <answer>'. Answer with a single word.
"""
      
llamaprompt = """\
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Keep your answer concise. Answer with a single word.

The answer is
"""

llamaCotPrompt = """
{story}
{question}
Choose from the following:
{containers_0}, {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.

Thought: Let's think step by step.
"""
