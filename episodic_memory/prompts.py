"""
Prompt templates for the Episodic Memory Module (RerankerAgent).

Ported from autogluon-assistant/src/autogluon/assistant/prompts/reranker_prompt.py.
"""

RERANKER_PROMPT = """\
Given the following context and list of tutorials with their summaries, select the {max_num_tutorials} most relevant tutorials for helping with this task. Consider how well each tutorial's title and summary match the task, data, user question, and any errors.

### Task Description
{task_description}

### Data Structures
{data_prompt}

### User Instruction
{user_input}

### Previous Error Analysis
{all_previous_error_analyses}

Available Tutorials:
{tutorials_info}

IMPORTANT: Respond ONLY with the numbers of the selected tutorials (up to {max_num_tutorials}) separated by commas. 
For example: "1,3,4" or "2,5" or just "1" if only one is relevant.
DO NOT include any other text, explanation, or formatting in your response.
"""
