from typing import List, Tuple, Dict
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from typing_extensions import Annotated, TypedDict
import re
from langchain_core.output_parsers import BaseOutputParser
import json
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from IPython.display import display, Latex
from langchain_core.output_parsers import StrOutputParser
from IPython.display import display, Latex

# model_name = "llama-3.3-70b-versatile"
model_name = "gpt-4.1"
#defining llm 

#llm for formulation
# llm_formulator = ChatOrlm(model_path = "./models/ORLM",max_new_tokens=1000000)
llm_formulator = ChatOpenAI(model=model_name, temperature = 0)
# llm_formulator = ChatGroq(model=model_name, temperature = 0)
from utils_files.FormOR import define_llm, ExplanationCritiqueOutput, LatexOutputParser

class LatexOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Match first LaTeX block between $$ with optional whitespace
        match = re.search(r'\$\$\s*([\s\S]*?)\s*\$\$', text)
        if match:
            return match.group(1).strip()
        # Fallback: Remove all non-LaTeX lines
        latex_lines = [line for line in text.split('\n') if line.strip().startswith(('\\', '$'))]
        return '\n'.join(latex_lines).strip()

critique_prompt = ChatPromptTemplate.from_template(
    """
    **ROLE**: You are an expert in Operations Research and Technical Writing.

    **TASK**: Given a problem description written in plain English, critique its **completeness**, **clarity**, and **technical accuracy**.

    **JSON Format Requirements**:
    - The output must be a JSON object with 4 keys:
         A **dictionary** with the following four keys, each containing a list of bullet points:
            - `Areas Lacking Clarity`: Identify any vague or ambiguous parts of the description.
            - `Suggestions for Improvement`: Recommend how to make the description clearer or more complete.
            - `Technical Gaps or Assumptions`: Highlight any important missing elements or implicit assumptions needed for modeling.
            - `Questions for the User`: Pose constructive questions that would help clarify or complete the problem.

    - Keep each list item short, actionable, and specific to the given problem.
    - If any assumption is clearly stated in the problem, do **not** flag it as missing.
    - Be polite, helpful, and technically rigorous.

    "**INPUT PROBLEM DESCRIPTION**"
    - Perform the above critique on:
    {problem_nl}
    """
) 
class critique_on_problem_dec:
    def __init__(self, problem, openai):
        # self.formulation = formulation
        self.llm = define_llm(openai)
        self.problem = problem
        
    def critique(self):
        chain = critique_prompt | self.llm 
        explanation = chain.invoke({
            # "formulation":self.formulation,
            "problem_nl":self.problem
        })
        explanation = explanation.content
        print(explanation)
        explanation = re.sub(r"^```json\n|```$", "", explanation.strip())
        parsed = json.loads(explanation)
        
        # Step 3: Normalize keys
        normalized = {
            "Areas Lacking Clarity": parsed.get("Areas Lacking Clarity", List),
            "Suggestions for Improvement": parsed.get("Suggestions for Improvement", List),
            "Technical Gaps or Assumptions": parsed.get("Technical Gaps or Assumptions", List),
            "Questions for the User": parsed.get("Questions for the User",List)
        }
        return normalized



explainer_prompt = ChatPromptTemplate.from_template(
    """
    **ROLE**: You are an expert in Operations Research with a focus on explaining and critiquing mathematical formulations.

    **JSON Format Requirements**:
    - The output must be a JSON object with two keys:
        1. `"EXPLANATION"`: A clear and concise explanation of the problem formulation written in **Markdown** format.
        2. `"CRITIQUE"`: A **list** of critique points (not paragraphs or Markdown sections) specifically relevant to the user's problem description and formulation.

    - The `"EXPLANATION"` section should include the following components as separate Markdown sections with appropriate headings:
        - ## Sets
        - ## Parameters
        - ## Decision Variables
        - ## Objective Function
        - ## Constraints

    - The `"CRITIQUE"` section should return a list of short, clear bullet points such as:
        - "The units for the decision variables are not defined."
        - "The objective function could clarify whether it's minimizing cost or maximizing profit."
        - "A few constraints could benefit from naming or annotation."

    **TASK**:
    - First, understand the context and intent of the given Operations Research problem formulation.
    - Then, explain each component of the formulation in a clear, structured, and educational manner using Markdown and LaTeX.
    - After that, critically evaluate the formulation, identifying any missing details, inconsistencies, ambiguities, or areas that could be improved. 
    - Ensure that the **CRITIQUE is specifically tailored to the context and contents of the given formulation**, not just general modeling issues.
    -If any assumption is stated in problem do not give critique on that
    -give critique on formulated constraint if it is formulated incorrectly

    "**INPUT FORMULATION**"
    - Perform the above task on: {formulation}
    - Problem Description: {problem_nl}
    """
)

class explanation_agent:
    def __init__(self, formulation,problem, openai):
        self.formulation = formulation
        self.llm = define_llm(openai)
        self.problem = problem
        
    def explain(self):
        chain = explainer_prompt | self.llm 
        explanation = chain.invoke({
            "formulation":self.formulation,
            "problem_nl":self.problem
        })
        explanation = explanation.content
        print(explanation)
        explanation = re.sub(r"^```json\n|```$", "", explanation.strip())
        parsed = json.loads(explanation)
        
        # Step 3: Normalize keys
        normalized = {
            "EXPLANATION": parsed.get("EXPLANATION", str),
            "CRITIQUE": parsed.get("CRITIQUE", str)
        }
        return normalized

correction_prompt = ChatPromptTemplate.from_template(
    "**ROLE**: You are expert in correcting formulation on given cririque on the existing formulation"

    "**TASK**: Correct the formulation and give Latex code enclosed in $$..$$"
    "Current Formulation: {formulation}"
    "Explanation Formulation: {explanation}"
    "Critique: {critique}"

    "FORMAT requirement"
    "give the formulation strictly in LaTeX enclosed between $$...$$"
)


class correction_agent:
    def __init__(self, formulation, exp_cri_dict, openai):
        self.formulation = formulation
        self.exp_cri_dict = exp_cri_dict
        self.llm = define_llm(openai)

    def correct(self):
        correction_chain = correction_prompt | self.llm | LatexOutputParser()
        corrected_formulation = correction_chain.invoke({"formulation": self.formulation,
                                "explanation":self.exp_cri_dict['EXPLANATION'],
                                "critique":self.exp_cri_dict['CRITIQUE']})
        return corrected_formulation

#code generation agent
code_generation_prompt = ChatPromptTemplate.from_template(
    """ROLE: You are a MASTER IN GUROBI PY CODING AND GIVEN THE FORMULATION YOU ARE TASKED TO CODE THE FORMULATION GIVEN TO YOU .
    if data is given to you then take that data 
    if data is not given to you then create data on your own which should be less and solvable by solver
    FORMULATION:
    {result}

    INSTRUCTIONS:
    1. Always define T from 0
    2. Print Decision varibles 
        if binary then only those with 1 
        

    FORMAT REQUIREMENTS:
    - Keep core model formulation intact
    - Use Python f-strings for dynamic parameter injection
    - Import correct libraries and modules 

    
    RETURN FULL PYTHON CODE IN ```..``` so that it will be easy to extract afterward.
    """
)

code_generation_formulation_chain = code_generation_prompt | llm_formulator | StrOutputParser()

def code_gen_agent(formulation):
    return code_generation_formulation_chain.invoke({"result":formulation})
