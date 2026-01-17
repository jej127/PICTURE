# Author: Shawn Lee
# Date: Aug 2023
# Description: File with simulation-related prompts for FANTOM.

PERSPECTIVE_PROMPT_LLAMA = """\
The following is a sequence of events about some characters, that takes place in multiple locations.
Your job is to output only the events that the specified character, {character}, knows about.

Story:
{story}

What events does {character} know about?"""

"""
1. A character knows about all utterances that they make.
2. If a character is in a certain room/location, that character knows about every utterance that occurs in the room.
3. If a character leaves a location, and is NOT in that location, they no longer know about any utterance that occurs within that location. However, they can re-enter the location.
"""

PERSPECTIVE_PROMPT= """\
The following is a sequence of utterances about some characters, that takes place in multiple locations.
Your job is to output only the utterances that the specified character, {character}, knows about.
1. A character knows about all utterances that they make.
2. If a character is in a certain room/location, that character knows about every utterance that occurs in the room.
3. If a character leaves a location, and is NOT in that location, they no longer know about any utterance that occurs within that location. However, they can re-enter the location.

Story:
{story}

What utterances does {character} know about? Only output the utterances according to the above rules, do not provide an explanation."""

PERSPECTIVE_PROMPT_OURS = """\
The following is a sequence of utterances about some characters, that takes place in multiple locations.
Here are a few rules:
1. A character knows about all utterances that they make.
2. If a character is in a certain room/location, that character knows about every utterance that occurs in the room.
3. If a character leaves a location, and is NOT in that location, they no longer know about any utterance that occurs within that location. However, they can re-enter the location.

Story:
{story}

The main question to be solved is as follows:

{question}

What events does and does not {character} know about? Output the answer for the main question by thinking step by step."""

PERSPECTIVE_PROMPT_OURS_LLAMA = """\
The following is a sequence of utterances about some characters, that takes place in multiple locations.
Here are a few rules:
1. A character knows about all utterances that they make.
2. If a character is in a certain room/location, that character knows about every utterance that occurs in the room.
3. If a character leaves a location, and is NOT in that location, they no longer know about any utterance that occurs within that location. However, they can re-enter the location.

Story:
{story}

The main question to be solved is as follows:

{question}

What events does and does not {character} know about? Output the answer for the main question by thinking step by step."""

PERSPECTIVE_PROMPT_DWM= """\
I give you a phrase of a dialogue between agents.
For each phrase, you must:
# 1. Write down a succinct description of what each agent knows about the environment and about the other agents. Keep the description short and do not produce redundant information.

Here's the dialogue:
{story}"""

PERSPECTIVE_PROMPT_PT="""\
Story:
{story}

Create a JSON array consisting of JSON objects. Each object should include an utterance from the dialogue and the audience for that utterance. Assume that characters in the story can hear every utterance that occurs while they are involved in the dialogue, but not those that occur when they are absent. Also, ensure that the speaker of each utterance is included in the audience.
Provide only the JSON array in the following format. Do not include any explanations.
[{{"Noah: Hi, Emma.": ["Noah", "Emma"]}},]"""

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

GPT_SIM_PROMPT_OURS = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
Keep your answer concise, one sentence is enough. You must choose one of the above choices.
"""

GPT_SIM_PROMPT2 = """\
{story}

You are {name}.
Based on the above information, answer the following question:
{question}

Keep your answer concise, one sentence is enough. You must choose one of the above choices.
{perspective}
"""

GPT_SIM_PROMPT_PRE = """\
{perspective}
{question}
Keep your answer concise, one sentence is enough. You must choose one of the above choices.
"""

LLAMA_SIM_PROMPT_PRE = """\
{perspective}
{question}
Keep your answer concise, one sentence is enough. You must choose one of the above choices.
"""

LLAMA_SIM_PROMPT2 = """\
{story}

You are {name}.
Based on the above information, answer the following question:
{question}

You must choose one of the above choices, do not say there is not enough information. Answer with a single word, do not output anything else.
{perspective}
"""

ABLATION_ONEPROMPT_SIM = """
I will give you a sequence of events about some characters, that takes place in multiple locations and a question that asks about the sequence of events. Your task is in two steps.

Here are a few rules:
    1. A character knows about all events that they do.
    2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
    3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

Story: {story}
Question: {question}
    
You will first output only the events that the specified character, {character}, knows about. You will then imagine you are the main character, {character}, then answer the question given to you based on the story you have rewritten. Do not output a blank answer or say you do not have enough information.
"""

######## Prompts for other methods ##########

baselinePrompt = """\
{story}
{question}
Choose from the following:
(a) {containers_0}, (b) {containers_1}
Keep your answer concise, one sentence is enough. You must choose one of the above choices.
"""

# baselinePrompt = """\
# {story}
# {question}
# Choose from the following:
# {containers_0}, {containers_1}
# """
                
questionPrompt = """\
{question}
Choose from the following:
(a) {containers_0}, (b) {containers_1}
""" 

questionPromptCoT = """\
{question}
Choose from the following:
(a) {containers_0}, (b) {containers_1}
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.
""" 
#Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: <answer>'. Answer with a single word.

questionPromptCoTdwm = """\
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
(a) {containers_0}, (b) {containers_1}
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

psPrompt = """
{story}
{question}
Choose from the following:
(a) {containers_0}, (b) {containers_1}
Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step. Write your final answer as 'Answer: <answer>'. Answer with a single word.
"""

oneShotPrompt = """
I will give you a story, then ask you to answer a question. For example:
1 James dislikes the suit.
2 James entered the staircase.
3 Noah entered the staircase.
4 The slacks is in the basket.
5 The basket is in the staircase.
6 Noah exited the staircase.
7 James moved the slacks to the treasure_chest.
8 The treasure_chest is in the staircase.
Where will Noah look for the slacks?
Answer: basket

Now, it is your turn.
{story}
{question}
Choose from the following:
{containers}
"""

oneShotCotPrompt = """
I will give you a story, then ask you to answer a question. For example:
1 James dislikes the suit.
2 James entered the staircase.
3 Noah entered the staircase.
4 The slacks is in the basket.
5 The basket is in the staircase.
6 Noah exited the staircase.
7 James moved the slacks to the treasure_chest.
8 The treasure_chest is in the staircase.
Where will Noah look for the slacks?
Noah sees that the slacks are in the basket, which is located in the staircase. Noah then leaves the staircase. James moves the slacks to the treasure_chest, but Noah does not see this as he is not in the staircase. Thus, Noah is not aware that the location of the slacks has been changed, and he still believes that the slacks are in the basket.
Thus, Noah will look for the slacks in the basket.
Answer: basket

Now, it is your turn.
{story}
{question}
Choose from the following:
{containers}
Think step by step, then output your answer at the end.
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


causePromptbase = """\
Read and comprehend the following story. Then, answer the question that follows.

{story}

{action} Would this behavior be different if {event}?
Answer the question with "Yes" or "No". Do not give any explanation.
"""

causePromptcot = """\
Read and comprehend the following story. Then, answer the question that follows.

{story}

{action} Would this result change if the event "{event}" did not happen?
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: Yes' or 'Answer: No'.
"""

causePromptbaserule = """\
Read and comprehend the following story. Then, answer the question that follows.

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

{story}

{action} Would this behavior be different if {event}?
Answer the question with "Yes" or "No". Do not give any explanation.
"""

causePromptcotrule = """\
Read and comprehend the following story. Then, answer the question that follows.

Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

{story}

{action} Would this behavior be different if {event}?
Reason step by step before answering in 'Thought: Let's think step by step'. Write your final answer as 'Answer: Yes' or 'Answer: No'.
"""

Ours_onePrompt = """
The following is a sequence of events about some characters, that takes place in multiple locations.
Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.

Story:
{story}

The main question to be solved is as follows:

{question}
Choose from the following:
{containers_0}, {containers_1}

What events does {character} know about? Output the answer for the main question by thinking step by step.
"""