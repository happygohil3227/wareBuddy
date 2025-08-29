from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from utils_files.code_corrector import CodeCorrector
from utils_files.utils import extract_python_code


# ══════════════════════════════════════════
#            Code Generation Chain
# ══════════════════════════════════════════
code_gen_prompt = PromptTemplate(
    input_variables=["formulation"],
    template="""
You are an expert in Python and metaheuristic optimization i.e. Genetic Algorithm

You are given an problem description and a mathematical formulation of the problem.

here is the problem description:
{problem_description}

Here is the problem formulation:
FORMULATION:
{formulation}
You are provided the following data objects already defined in memory:
{given_data_set_description}

Your task:
1. representation of initial population 
   {initial_pop_description}
2. Cross over 
   {cross_over_description}
3. Mutation
   {mutation_description}
4. fitness function
   {fitness_description}
5. Selection
   {selection_description}
6. look for edge cases in every block of code and add it in 
4. Implement the Genetic algorithm using **only** built-in Python modules, **NumPy**, and **Matplotlib**.
5. Include **automatic hyperparameter tuning** within the code. Loop over options like population sizes, mutation/cooling rates, generations, turnament size, and compare performance.
while finetuning hyperparameters, ensure that you print only the best cost for each hyperparameter combination.
6. population size should be 1000, turnament size should be 10, crossover rate should be 0.8, mutation rate should be 0.1, generations should be 1000
7. Track and plot the **convergence history** (e.g., best total distance per iteration).
8. Ensure that constraints (such as fulfilling `rp` demand and valid location visits) are respected. Apply penalties for infeasible solutions in the fitness function.
9. At the end, print and return:
   - The best solution found (clearly, with structure and interpretation)
   - The final tuned hyperparameters
   - The convergence plot
10. do not provide if __name__ == "__main__" block
11. Ensure the code is syntactically correct and can be executed directly in a Python environment.

Only output a **complete runnable Python script** enclosed in triple backticks (```), and nothing else.
**Do NOT generate fake or random data. Use the `q_pi`, `rp`,`distance_matrix`, `product`, `locations` and `location_start` variables as provided.**

"""
)

# LLM
llm = ChatOpenAI(temperature=0, model="gpt-4")

# LLMChain
metaheuristic_chain = code_gen_prompt | llm 


from io import StringIO
import sys
import traceback
# ══════════════════════════════════════════
#            Code execution function
# ══════════════════════════════════════════
def execute_code_string(code_string, global_vars=None):
    """
    Executes a code string and returns a dict with:
    result 
    - 'status': True if successful, False if error occurred
    - 'output': printed output or error traceback
    """
    if global_vars is None:
        global_vars = globals()

    output_stream = StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = output_stream
        exec(code_string, global_vars)
        status = True
    except Exception:
        traceback.print_exc(file=output_stream)
        status = False
    finally:
        sys.stdout = original_stdout

    output = output_stream.getvalue()
    output_stream.close()

    return {
        "status": status,
        "output": output
    }


# ══════════════════════════════════════════
#            Code correction agent
# ══════════════════════════════════════════
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# ---------- feedback prompt template ----------
feedback_prompt = PromptTemplate(
    input_variables=["code", "error"],
    template="""
You are an expert Python programmer and code reviewer.

Below is a Python script that failed with an error. 
You are provided the following data objects already defined in memory:
- `q_pi`: A pandas DataFrame with product IDs as rows and location IDs as columns. `q_pi.loc[p][l]` gives the quantity of product `p` available at location `l`.
- `rp`: dict with product IDs as keys and required quantities as values. `rp[p]` gives the total quantity required for product `p`.
- `distance_matrix`: A square DataFrame with location IDs as both rows and columns, representing the distance between locations.
- `products`: A list of product IDs that need to be picked.
-`locations`: A list of location IDs where products can be picked.
-`location_start`: str The starting location for the picking route. 


--- CODE ---
{code}
--- ERROR ---
{error}
for your reference only Here is the problem and its formulation for which this code of meta-huristic {algoName} is written
----Problem Description----
{problem}
----formulation----
{formulation}
--- END ---

Please analyze the code and the error message, then provide a reasoning on why code is failing and how to fix it.
"""
)

llm = ChatOpenAI(temperature=0, model="gpt-4.1")
feedback_chain = feedback_prompt | llm

# ---------- Correction chain----------
correction_prompt = PromptTemplate(
    input_variables=["code", "error", "feedback"],
    template="""
You are an expert Python programmer and code reviewer.



You are provided the following data objects already defined in memory:
- `q_pi`: A pandas DataFrame with product IDs as rows and location IDs as columns. `q_pi.loc[p][l]` gives the quantity of product `p` available at location `l`.
- `rp`: dict with product IDs as keys and required quantities as values. `rp[p]` gives the total quantity required for product `p`.
- `distance_matrix`: A square DataFrame with location IDs as both rows and columns, representing the distance between locations.
- `products`: A list of product IDs that need to be picked.
-`locations`: A list of location IDs where products can be picked.
-`location_start`: str The starting location for the picking route. 

Below is a Python script that failed with an error & feedback on how to fix it.
--- CODE ---
{code}
--- ERROR ---
{error}
----FEEDBACK---
{feedback}

Here is the problem and its formulation for which this code of meta-huristic {algoName} is written
----Problem Description----
{problem}
----formulation----
{formulation}
--- END ---

Please analyze the code, the error message, and the feedback provided, then rewrite the code to fix the error.
and lookafter the end result should not be hampered while correcting the error
Make sure the new code is runnable and corrects the issues mentioned in the feedback.
"""
)

llm = ChatOpenAI(temperature=0, model="gpt-4.1")
correction_chain = correction_prompt | llm

from langchain.chains import SequentialChain


code_correction_chain = feedback_chain | correction_chain

import sys
from io import StringIO

# ══════════════════════════════════════════
#            Code correction agent 
# ══════════════════════════════════════════
def run_code_correction_agent(code, code_correction_chain,exec_globals, max_iterations=3):
    """_summary_

    Args:
        code (str): _description_
        code_correction_chain (RunnableChain): _description_
        exec_globals (dict): _description_
        max_iterations (int, optional): _description_. Defaults to 3.

    Returns:
        tuple:
        code: final code 
        result: output of the final code
    """
    # Step 1: Generate initial code
    for iteration in range(max_iterations):
            result = execute_code_string(code, exec_globals)
            if result["status"]:
                print(f"Success on iteration {iteration+1}")
            else:
                # Use the code correction chain to fix the code
                # print(code)
                feedback_response = feedback_chain.invoke({'code': code, 'error': result['output']})
                feedback = feedback_response.content 
                print(f"Feedback received: {feedback}")
                correction_response = code_correction_chain.invoke({
                    'code': code,
                    'error': result['output'],
                    'feedback': feedback
                })
                code = correction_response.content 
                code = extract_python_code(code )
    return code, result['output']

