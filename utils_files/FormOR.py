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
# model_name = "gpt-4.1"
#defining llm 

#llm for formulation
# llm_formulator = ChatOrlm(model_path = "./models/ORLM",max_new_tokens=1000000)
# llm_formulator = ChatOpenAI(model=model_name, temperature = 0)
# llm_formulator = ChatGroq(model=model_name, temperature = 0)

def define_llm(openai):
    if openai:
        return ChatOpenAI(model ="gpt-4.1",temperature = 0)
    else:
        return ChatGroq(model="llama-3.3-70b-versatile", temperature = 0)

class ProblemClauses(TypedDict):
    sets: Annotated[List[str], "List of sets used in the formulation"]
    parameters: Annotated[List[str], "List of parameters used in the formulation"]
    decision_variables: Annotated[List[str], "List of decision variables used in the formulation"]
    objective_clause: Annotated[str, "Description of the objective to be optimized"]
    constraint_clauses: Annotated[List[str], "List of constraints applied in the problem"]

class ExplanationCritiqueOutput(TypedDict):
    EXPLANATION: Annotated[str, "Markdown-formatted explanation of the formulation including sets, parameters, decision variables, objective function, and constraints"]
    CRITIQUE: Annotated[str, "Markdown-formatted critique of the formulation including issues, assumptions, and suggestions for improvement"]

class LatexOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Match first LaTeX block between $$ with optional whitespace
        match = re.search(r'\$\$\s*([\s\S]*?)\s*\$\$', text)
        if match:
            return match.group(1).strip()
        # Fallback: Remove all non-LaTeX lines
        latex_lines = [line for line in text.split('\n') if line.strip().startswith(('\\', '$'))]
        return '\n'.join(latex_lines).strip()


clause_extractor_CoT_prompt = ChatPromptTemplate.from_template(
    "Given an Operations Research problem in natural language, you have to think step-by-step and extract the following structured components:\n"
    "- sets\n"
    "- parameters\n"
    "- decision variables\n"
    "- objective clause\n"
    "- constraint clauses\n\n"

    "Please analyze the problem step-by-step and finally respond in the following JSON/dict format:\n"
    "{{set: [<set_clauses>], parameter: [<parameter_clauses>], Decision var: [<decision_variable_clauses>], objective: <objective_clause>, Constraints: [<list_of_constraints>]}}\n\n"
    "For Example:\n"
    "Problem: A manufacturing company produces n different products. Each product requires a specific amount of machine time and labor hours to manufacture. The company has a limited number of machine hours and labor hours available each week. Additionally, each product earns a certain profit per unit sold. The goal of the company is to determine how many units of each product it should produce in order to maximize total profit, without exceeding the available machine time and labor hours. The production of each product must also be non-negative, meaning the company cannot produce a negative quantity of any product.\n\n"
    "Answer:\n"
    "Step-by-Step Reasoning\n\n"
    "1. Identify the Set(s)"
"We are dealing with n different products. Each product has its own requirements and profit."

"So, we define:"
"Let P be the set of all products, indexed by i ∈ P"

"2. Identify the Parameters"
"From the problem, for each product i ∈ P, we are told:"
"- It requires a specific amount of machine time."
"- It requires a specific amount of labor hours."
"- It earns a certain profit per unit sold."

"Also, there are total available:"
"- Machine hours"
"- Labor hours"

"So, we define the following parameters:"
"- m_i: machine hours required to produce one unit of product i"
"- l_i: labor hours required to produce one unit of product i"
"- p_i: profit per unit of product i"
"- M: total machine hours available"
"- L: total labor hours available"

"3. Identify the Decision Variables"
"We are asked to determine how many units of each product to produce."

"So, we define:"
"- x_i: number of units of product i to produce, for all i ∈ P"

"4. Define the Objective Clause"
"The company wants to maximize total profit. Profit is calculated as the number of units of a product multiplied by the profit per unit."

"Objective clause:"
"- Maximize the sum of profits over all products."

"5. Define the Constraint Clauses"
"From the description:"
"- The total machine time used cannot exceed the available machine time M"
"- The total labor hours used cannot exceed the available labor hours L"
"- The production quantity of any product cannot be negative"

"Constraint clauses:"
"- Machine time constraint: Total machine time used (sum over all products) ≤ M"
"- Labor time constraint: Total labor time used (sum over all products) ≤ L"
"- Non-negativity constraint: x_i ≥ 0 for all i ∈ P"
    "Final Output:\n"
    """{{
  set: [
    "P: set of products (indexed by i)"
  ],
  parameter: [
    "m_i: machine hours required per unit of product i",
    "l_i: labor hours required per unit of product i",
    "p_i: profit per unit of product i",
    "M: total available machine hours",
    "L: total available labor hours"
  ],
  Decision var: [
    "x_i: number of units of product i to produce, for all i ∈ P"
  ],
  objective: "Maximize total profit = sum over i of (p_i * x_i)",
  Constraints: [
    "The total machine time used cannot exceed the available machine time M",
    "The total labor hours used cannot exceed the available labor hours L",
    "The production quantity of any product cannot be negative"
  ]
}}"""
    "Now extract the components from the following problem:"
    "Now analyze the following problem:\n"
    "Problem: {problem_description}"
)

clause_extractor_CoT_prompt_2 = ChatPromptTemplate.from_template(
    "**ROLE**: Extract the cluases of the OR problem from reasoning done by LLM"

    "**TASK**"
    "you have to extract from the give input only JSON/dict format:\n"
    "{{\"set\": [<set_clauses>], \"parameter\": [<parameter_clauses>], \"Decision var\": [<decision_variable_clauses>], \"objective\": <objective_clause>, \"Constraints\": [<list_of_constraints>]}}\n\n"

    "**USER INPUT**"
    "Reasoning: {reason}"
)



class cluase_extractor:
    def __init__(self, problem_description, openai):
        self.problem_description = problem_description
        self.llm = define_llm(openai)

    def run(self):
        llm_clause_resoaner = clause_extractor_CoT_prompt | self.llm | StrOutputParser()
        reasoning = llm_clause_resoaner.invoke({"problem_description":self.problem_description})
        llm_chain = clause_extractor_CoT_prompt_2 | self.llm | StrOutputParser()
        final = llm_chain.invoke({"reason":reasoning})
    
        # Step 1: Strip markdown-style backticks
        clean_json_str = re.sub(r"^```json\n|```$", "", final.strip())
        
        # Step 2: Parse JSON
        parsed = json.loads(clean_json_str)
        
        # Step 3: Normalize keys
        normalized = {
            "sets": parsed.get("set", []),
            "parameters": parsed.get("parameter", []),
            "decision_variables": parsed.get("Decision var", []),
            "objective_clause": parsed.get("objective", ""),
            "constraint_clauses": parsed.get("Constraints", [])
        }

        return normalized
        

        

constraint_prompt = ChatPromptTemplate.from_template(
    """  
    Given the problem, Parameter, Decision variable, Objective of the optimization problem
    Problem description:{problem_dics}
    Sets: {sets}
    Paramter description: {parameter_dics}
    decision variable description: {decision_var_disc}
    objective function description: {objective_disc}
    you have to formulate the {constraint} which will be one constraint equation not more then one 
    
    Please think step by step on formulating only one constraint
    For example(This way you have to think): 
    backlog update: it should update bjkt , when j is selected(wj(t-1)=1) we will add sjkt which is that to be picked in next time periods
    Thought: bjkt need to be updated over time bjkt is updated only when shipment j is selected at t-1 and bjkt should be updated using the sjkt which is units of k not picked for shipment j at time t
    and there should be no sumation over any variable as we are updating bjkt which is single variable so the constraint will be bjkt = bjkt-1 (1-wjt-1) + sjkt-1

    and finally give the Latex code of the constraints
    """
)

constraint_extract_prompt = ChatPromptTemplate.from_template(
    "given the reasoning of the constraint formulation: {reasoning}"
    "give the final constraint as Latex code"
    "give the constraint strictly in LaTeX enclosed between $$...$$"
    
)


objective_prompt = ChatPromptTemplate.from_template(
    """  
    Given the problem, Parameter, Decision variable, Objective of the optimization problem
    Problem description:{problem_dics}
    Paramter description: {parameter_dics}
    decision variable description: {decision_var_disc}
    you have to formulate the {Objective} which will be one Objective equation not more then one 
    
    Please think step by step on Formulation of the objective function
    
    For example(This way you have to think): 
    Objective: Minimize the total travel distance 
    Thought: Travel distance is the distance between two locations and here total travel distance is given. In VRP dij.xij is the distance travelled if xij =1 i.e. ij edge is chosen and here it is total so dij.xij should be sum over i & j
            finally objective will me sum(i,j) xij dij i, j belong to set of location

    and finally give the Latex code of the objective function
    """
)

objective_extract_prompt = ChatPromptTemplate.from_template(
    "given the reasoning of the objective formulation: {reasoning}"
    "give the final objective function as Latex code"
    "give the Objective function strictly in LaTeX enclosed between $$...$$"
    
)


class formulation_objective_constraints:
    def __init__(self, problem_description, problem_dict, openai):
        self.problem_description = problem_description
        self.problem_dict = problem_dict
        self.llm = define_llm(openai)

    def run(self):
        #constraints 
        constraint_chain = constraint_prompt | self.llm | StrOutputParser()
        constraint_list = []
        problem_dict = self.problem_dict
        # print(problem_dict)
        for cons in problem_dict["constraint_clauses"]:
            constraint  = constraint_chain.invoke({"constraint": cons,
                                    "sets":problem_dict["sets"][0],
                                    "problem_dics": self.problem_description,
                                    "parameter_dics": " ".join(problem_dict["parameters"]),
                                    "decision_var_disc": " ".join(problem_dict["decision_variables"]),
                                    "objective_disc": problem_dict["objective_clause"]})
            constraint_list.append(constraint)

        constraint_extract_chain = constraint_extract_prompt | self.llm | LatexOutputParser()
        
        constraint_list_equation = []
        for constraint in constraint_list:
            con = constraint_extract_chain.invoke({"reasoning":constraint})
            constraint_list_equation.append(con)

        #objective
        objective_chain = objective_prompt | self.llm | StrOutputParser()
        objective_extraction_chain = objective_extract_prompt | self.llm | LatexOutputParser()
        obective = objective_chain.invoke({"Objective": problem_dict["objective_clause"],
                            "problem_dics": self.problem_description,
                            "parameter_dics": problem_dict["parameters"],
                            "decision_var_disc": problem_dict["decision_variables"]})
        
        objective = objective_extraction_chain.invoke({"reasoning":obective})

        return constraint_list_equation, objective

parameter_dec_var_prompt = ChatPromptTemplate.from_template(
    """**ROLE**: You are a LaTeX code generator for formulating Operations Research (OR) problems.

**TASK**: Given the components of an OR problem (sets, parameters, decision variables, objective, constraints), write clean and complete LaTeX code for the mathematical formulation.

**INPUT**:
- Sets: {set}
- Parameters: {parameters}
- Decision Variables: {decision_var}
- Objective: {objective}
- Constraints: {constraints}

**OUTPUT FORMAT REQUIREMENTS**:
Respond only with LaTeX code enclosed within `$$` so it can be rendered using `display(Latex(...))` in Python.

The structure of the output should follow this order:
1. **Sets**
2. **Parameters**
3. **Decision Variables**
4. **Objective Function**
5. **Constraints**

Wrap each section in appropriate LaTeX math mode, using `\\` for line breaks if necessary.

**OUTPUT**:
$$
\\textbf{{Sets:}} \\\\
{{LaTeX sets formulation}}

\\textbf{{Parameters:}} \\\\
{{LaTeX parameters formulation}}

\\textbf{{Decision Variables:}} \\\\
{{LaTeX decision variables formulation}}

\\textbf{{Objective Function:}} \\\\
{{LaTeX objective function}}

\\textbf{{Constraints:}} \\\\
{{LaTeX constraints formulation}}
$$
"""
)


class formulation:
    def __init__(self, problem_dict, con, objective, openai):
        self.problem_dict = problem_dict
        self.llm = define_llm(openai)
        self.con = con
        self.objective = objective

    def run(self):
        # problem_dict = self.problem_dict
        constraints = ' '.join(self.con)
        formulation_chain = parameter_dec_var_prompt | self.llm | LatexOutputParser()
        formulation = formulation_chain.invoke(
            {
                "set":self.problem_dict['sets'],
                "parameters":self.problem_dict['parameters'],
                "decision_var": self.problem_dict['decision_variables'],
                "objective": self.objective,
                "constraints": constraints
            }
        )
        self.formulation = formulation
        return formulation

    def display_formulation(self):
        display(Latex(self.formulation))
        
def load_sample_problem():
    '''
    loading the problem from txt file
    '''
    def read_file(file_path):
        with open(file_path, "r") as f:
            return f.read() 

    def parse_constraints(text):
        # Extract constraints starting with numbered patterns (e.g., "1.", "2.", etc.)
        matches = re.split(r'\n\s*(\d+)\.\s*', text.strip())
        constraints = []
        if len(matches) > 1:
            # Combine numbers and following lines
            for i in range(1, len(matches), 2):
                constraint = matches[i+1].strip()
                constraints.append(f"{matches[i]}. {constraint}")
        else:
            # Fallback: split by newlines
            constraints = [line.strip() for line in text.split("\n") if line.strip()]
        return constraints

    sample_problem = {
        "problem_disc": read_file("Problem/problem_description.txt"),
        "parameter_disc": read_file("Problem/parameter_description.txt"),
        "decision_var_disc": read_file("Problem/decision_variable_description.txt"),
        "objective_disc": read_file("Problem/objective_description.txt").strip(),
        "constraint_disc": parse_constraints(read_file("Problem/constraint_description.txt"))
    }
    return sample_problem






