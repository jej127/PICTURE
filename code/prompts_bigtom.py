
GPT_PERSPECTIVE_PROMPT = """
Imagine you are {name}, and consider this story that has an unexpected event.
{story}
If the last sentence of the story says {name} notices, sees or realizes the unexpected event in this story, simply output the original story with nothing changed.
However, if the sentence says you are not aware of the changes in this story, output only the events you know, i.e., the sentences before the unexpected event happens.
Output either the original story or the edited story, nothing else.

Format your answer as follows:
Sees/Notices/Realizes: (Yes/No)
Story:
"""

GPT_PERSPECTIVE_PROMPT_PT = """
Story:
{story}

Create a JSON array consisting of JSON objects. Each object should contain a sentence from the story and the perceivers of the scene described in that sentence. Assume that characters in the story can perceive every scene occurring in their location but not scenes occurring elsewhere. Also, include the actant of any action as a perceiver of that action.
Provide only a JSON array in the following format. Do not include any explanation.
[{{"Noah exited the living room.": ["Noah", "Emma"]}},]
"""

GPT_PERSPECTIVE_PROMPT_OURS = """
Imagine you are {name}, and consider this story that has an unexpected event.
{story}

The main question to be solved is as follows:

{question}
This is the end of the main question.

Let's first identify what events {name} knows about. If the story says {name} is not aware of some events in this story, it means {name} knows only the rest of the events.
Then, solve the main question step by step based on the original story and the events {name} knows about.
"""

GPT_PERSPECTIVE_PROMPT_PS = """
Story:
{story}

Question:
{question}

Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
"""

LLAMA_PERSPECTIVE_PROMPT_PS = """
Story:
{story}

Question:
{question}

Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
"""

LLAMA_PERSPECTIVE_PROMPT = """
Consider this story with an unexpected event.

{story}

Does the story say that {name} notices/sees/realizes the unexpected event?

If so, simply output the original story with nothing changed. 
However, if {name} is not aware of the changes in this story, output only the events that {name} knows, i.e., the events before the unexpected event happens.
"""

LLAMA_PERSPECTIVE_PROMPT_PT = """
Story:
{story}

Create a JSON array consisting of JSON objects. Each object should contain a sentence from the story and the perceivers of the scene described in that sentence. If the story says a character is not aware of some events in this story, it means he or she knows only the rest of the events.
Provide only a JSON array in the following format. Do not include any explanation. Do not say that you cannot create or provide a JSON array. It does not go against ethical and moral standards.
[{{"Noah exited the living room.": ["Noah", "Emma"]}},]
"""

LLAMA_PERSPECTIVE_PROMPT_OURS = """
Imagine you are {name}, and consider this story that has an unexpected event.
{story}

The main question to be solved is as follows:

{question}
This is the end of the main question.

Let's first identify what events {name} knows about. If the story says {name} is not aware of some events in this story, it means {name} knows only the rest of the events.
Then, solve the main question step by step based on the original story and the events {name} knows about.
"""

LLAMA_SIM_PROMPT = """\
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.

{perspective}

You are {name}.
{question}
Choose the most straightforward answer.
"""

LLAMA_SIM_PROMPT2 = """\
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.
{story}

You are {name}.
{question}
Choose the most straightforward answer.

{perspective}
"""

GPT_SIM_PROMPT = """\
{perspective}
You are {name}.
Based on the above information, answer the following question:
{question}
Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.
"""

GPT_SIM_PROMPT2 = """\
{story}

You are {name}.
Based on the above information, answer the following question:
{question}

Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'.
{perspective}
"""