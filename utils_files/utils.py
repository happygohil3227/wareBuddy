import os
import re
import subprocess
import time
from datetime import datetime
import pytz

from typing import Annotated
from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages

# LangChain core modules
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.chat_history import ChatMessageHistory

# LangChain community and provider models
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama, ChatHuggingFace
from langchain_groq import ChatGroq

# Memory, prompts, and chains
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain


import re
from langchain_core.output_parsers import BaseOutputParser
import pandas as pd

# def data_dec_gen(df: pd.Dataframe):
#     prompt = ChatPromptTemplate.from_template(
#     """
#         *ROLE*: you are great in explaining the provided data
#         *TASK*: Your task is to iterpreat the data from given infomation of the data and provide a description that can be used in downstream process of problem solving

#         *INPUT*
#     """ 
#     )

class LatexOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Match first LaTeX block between $$ with optional whitespace
        match = re.search(r'\$\$\s*([\s\S]*?)\s*\$\$', text)
        if match:
            return match.group(1).strip()
        # Fallback: Remove all non-LaTeX lines
        latex_lines = [line for line in text.split('\n') if line.strip().startswith(('\\', '$'))]
        return '\n'.join(latex_lines).strip()


def create_custom_agent(system_prompt, model_name, provider, temperature=0.7, verbose=False):
    # Step 1: Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{input}")
    ])
    
    # Step 2: Choose model
    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "groq":
        llm = ChatGroq(model_name=model_name, temperature=temperature)
    elif provider == "huggingface":
        llm = ChatHuggingFace(model_name=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Step 3: Combine prompt and LLM into a chain
    chain = prompt | llm

    # Step 4: Create Runnable with memory
    memory = ConversationBufferMemory(return_messages=True)
    message_history = lambda session_id: memory.chat_memory

    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history=message_history,
        input_messages_key="input",  # this must match your prompt input
        history_messages_key="history"
    )

    return runnable

def create_custom_agent2(system_prompt, model_name, provider, temperature=0.7):
    # 1. Define State Structure
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # 2. Create Graph Builder with State
    graph_builder = StateGraph(AgentState)

    # 3. Add LLM Node
    def llm_node(state: AgentState):
        # Initialize model based on provider
        if provider == "openai":
            llm = ChatOpenAI(model=model_name, temperature=temperature)
        elif provider == "groq":
            llm = ChatGroq(model_name=model_name, temperature=temperature)
        elif provider == "huggingface":
            llm = ChatHuggingFace(model_name=model_name)
        
        # Build prompt with current messages
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content="{input}")
        ])
        
        response = llm.invoke(prompt.format_messages(input=state["messages"][-1].content))
        return {"messages": [response]}

    # 4. Configure Graph
    graph_builder.add_node("llm", llm_node)
    graph_builder.add_edge(START, "llm")
    graph_builder.add_edge("llm", END)

    # 5. Add Memory Support
    memory = MemorySaver()  # LangGraph's recommended checkpointer
    agent = graph_builder.compile(checkpointer=memory)

    return agent


# def create_custom_agent(
#     system_prompt: str,
#     model_name: str,
#     provider: str = "openai",  # 'openai', 'ollama', 'groq'
#     temperature: float = 0.7,
#     verbose: bool = False
    
# ):
#     """
#     Creates a conversational agent using LangChain with support for multiple LLM providers.

#     Args:
#         system_prompt (str): Instruction for the system message.
#         model_name (str): Model name like 'gpt-4', 'llama2', 'mixtral-8x7b', etc.
#         provider (str): LLM provider - 'openai', 'ollama', or 'groq'.
#         temperature (float): Creativity level of the model.
#         verbose (bool): Whether to enable verbose output.

#     Returns:
#         ConversationChain: A configured LangChain agent.
#     """

#     # Select LLM provider
#     if provider == "openai":
#         llm = ChatOpenAI(model_name=model_name, temperature=temperature)
#     elif provider == "ollama":
#         llm = ChatOllama(model=model_name, temperature=temperature)
#     elif provider == "groq":
#         llm = ChatGroq(model_name=model_name, temperature=temperature)
#     else:
#         raise ValueError(f"Unsupported provider: {provider}")

#     # Create memory to retain chat history
#     memory = ConversationBufferMemory(return_messages=True)

#     # Use a ChatPromptTemplate for consistent formatting
#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessage(content=system_prompt),
#         MessagesPlaceholder(variable_name="history"),
#         HumanMessage(content="{input}")
#     ])

#     # Build the conversation chain
#     chain = ConversationChain(
#         memory=memory,
#         prompt=prompt,
#         llm=llm,
#         verbose=verbose
#     )
#     from langchain_core.runnables import RunnableWithMessageHistory

#     runnable = RunnableWithMessageHistory(
#         chain, 
#         get_session_history=..., 
#         input_messages_key="input",
#         history_messages_key="history"
#     )


#     return chain


from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain

def create_custom_agent_orlm(
    system_prompt: str,
    model_name: str,
    temperature: float = 0.7,
    huggingface_api_key: str = None,
    verbose: bool = False
):
    """
    Create a conversational agent using Hugging Face model via LangChain.

    Args:
        system_prompt (str): Instruction for the agent's role.
        model_name (str): Hugging Face model repo name (e.g., 'HuggingFaceH4/zephyr-7b-beta').
        temperature (float): Response randomness.
        huggingface_api_key (str): Hugging Face Hub API key.
        verbose (bool): Enable verbose output.

    Returns:
        ConversationChain: Configured agent using Hugging Face model.
    """

    if huggingface_api_key:
        import os
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

    # Create chat model using HuggingFaceHub
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={"temperature": temperature, "max_new_tokens": 512}
    )

    # For non-chat models, you'll use a slightly different flow (see below if needed)

    # Memory for maintaining history
    memory = ConversationBufferMemory(return_messages=True)

    # Prompt setup with system message
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{input}")
    ])

    # Build the conversational chain
    chain = ConversationChain(
        memory=memory,
        llm=llm,
        prompt=prompt,
        verbose=verbose
    )

    return chain




def ask_llm_langchain_groq(llm_model, messages):
    # start_time = time.time()
    # cur_time = datetime.now(LA_TIMEZONE)
    # print(f"[{cur_time.strftime('%Y-%m-%d %H:%M:%S')}]\tAsking LLM (LangChain-Groq) ...")

    # Create Groq LLM
    
    # llm = ChatGroq(model=llm_model)
    llm = ChatOllama(model=llm_model, temprature=0.5)

    # Convert OpenAI-style dict messages to LangChain Message objects
    lc_messages = []
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # Get completion
    response = llm.invoke(lc_messages)

    end_time = time.time()
    # response_time = end_time - start_time
    # cur_time = datetime.now(LA_TIMEZONE)
    # print(f"[{cur_time.strftime('%Y-%m-%d %H:%M:%S')}]\tFinished.\tResponse time: {response_time:.2f} seconds.")

    response_content = response.content

    # You may still use token counting (if you have a custom function or use tiktoken)
    # calculate_token_num(messages=messages, response_content=response_content,
    #                     model=llm_model, token_file_path=token_file_path, notes=notes)

    return response_content

def refine_code_with_error(original_code, error_message,llm_model):
    prompt = f"""
You are a Python code fixer. You will be given a Python code and an error message. Your task is to fix the code based on the error message.

{error_message}

Please fix the bug and return corrected, complete working Python code.

Original code:
{original_code}
START YOUR CODE IN THIS FORMATE Python```...``` Stick to this formate 

INCORRECT OUTPUT FORMAT:
```python from ortools.cons...'''

please generate ONLY def solve_vrp(data: VehicleRoutingProblem) -> tuple[list[list[int]], float]: function to solve the problem.


"""

    messages = [
    {"role": "system", "content": "Your are python code fixer and will give full corrected code"},
    {"role": "user", "content": prompt},
    ]

    # fixed_code = ask_llm_langchain_groq(llm_model, messages)
    response_content= ask_llm_langchain_groq(llm_model, messages)
    code = extract_python_code(response_content)
    # cleaned_code = re.sub(r"```(?:python)?", "", fixed_code)
    # cleaned_code = re.sub(r"```", "", cleaned_code).strip()
    return code

def extract_python_code(content):
    pattern = re.compile(r'```(?:python)?(.*?)```', re.DOTALL)
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    else:
        tmp_str = 'print("No Python code block found in the solution reply.")'
        print(tmp_str)
        return tmp_str


def check_log_file_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        output_section = content.split('OUTPUT:')[1].split('ERROR:')[0].strip()
        error_section = content.split('ERROR:')[1].strip()

        # Check if the sections are empty
        output_empty = (output_section == "")
        error_empty = (error_section == "")

    # Determine the result based on the conditions
    if output_empty and error_empty:
        return 'EMPTY'
    else:
        return 'NOT EMPTY'

def read_txt_file(path):
    with open(path, 'r') as file:
        content = file.read()

    return content

def write_py_file(file_name, content):
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Combine the current directory with the file name
    file_path = os.path.join(current_dir, file_name)

    # Write content to the file
    with open(file_path, 'w') as file:
        file.write(content)

def run_py_file(code_path, log_path, max_exec_time):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Execute the Python file
    timeout_seconds = max_exec_time

    start_time = time.time()
    # cur_time = datetime.now(LA_TIMEZONE)
    # print(f"[{cur_time.strftime('%Y-%m-%d %H:%M:%S')}]\tExecution {code_path}, Log: {log_path}, Max Time: {max_exec_time}...")

    try:
        result = subprocess.run(['python', code_path], capture_output=True, text=True, timeout=timeout_seconds)
        exec_success = (result.returncode == 0)
        if exec_success:
            log_path = log_path.replace('.txt', '_success.txt')
            exec_status_str = 'success'
        else:
            log_path = log_path.replace('.txt', '_error.txt')
            exec_status_str = 'error'

        # Prepare log information
        log_info = {
            'output': result.stdout,
            'error': result.stderr
        }

    except subprocess.TimeoutExpired as _:
        log_path = log_path.replace('.txt', '_timeout.txt')
        exec_status_str = 'timeout'
        log_info = {
            'output': '',
            # 'error': f'TimeoutExpired: The process exceeded the time limit of {timeout_seconds} seconds.'
        }

    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    # cur_time = datetime.now(LA_TIMEZONE)
    # print(f"[{cur_time.strftime('%Y-%m-%d %H:%M:%S')}]\tFinished.\tExecution time: {execution_time:.2f} seconds.")

    # Save log information to a file
    with open(log_path, 'w') as log_file:
        for key, value in log_info.items():
            log_file.write(f"{key.upper()}:\n{value}\n\n")

    return exec_status_str, execution_time, log_path


from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
import nest_asyncio

# Required for Jupyter Notebook async handling
nest_asyncio.apply()
def visualize_graph_locally(app):
    try:
        display(Image(app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(
                first="#ffdfba", 
                last="#baffc9", 
                default="#fad7de"
            ),
            wrap_label_n_words=9,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=5,
            max_retries=3,
            retry_delay=2.0
        )))
    except ImportError:
        print("Please install required packages:")
        print("!pip install pyppeteer nest_asyncio")

from IPython import get_ipython
def write_py_code_next_cell(code):
    ip = get_ipython()
    ip.set_next_input(extract_python_code(code), replace=False)


