# code_corrector.py
from typing import List, Tuple, Dict
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict
from langchain_core.output_parsers import BaseOutputParser
import re
# from utils import extract_python_code

class CodeCorrector:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name)
        self._initialize_components()
        self._build_graph()

    def _initialize_components(self):
        # Define schemas and prompts
        class CorrectionFeedback(TypedDict):
            correction_needed: Annotated[bool, ..., "Whether correction is needed in code?"]
            correction: Annotated[str, ..., "If needed, which line has issue and suggested change"]

        self.CorrectionFeedback = CorrectionFeedback

        self.correction_feedback_prompt = ChatPromptTemplate.from_template(
            "Analyze this code and error according to the correction request.\n\n"
            "Code:\n{code}\n\n"
            "Error:\n{error}\n\n"
            "Correction request: {correction_request}\n\n"
            "If no correction needed, state why. Otherwise, specify line numbers and fixes."
        )

        self.revision_prompt = ChatPromptTemplate.from_template(
            "Revise this code based on the feedback.\n\n"
            "inside triple backticks (```...```)"
            "Original Code:\n{code}\n\n"
            "Error:\n{error}\n\n"
            "Feedback: {feedback}\n\n"
            "Correction Request: {correction_request}\n\n"
            "If feedback doesn't require changes, return original code. "
            "Otherwise, provide corrected code with explanations."
        )

        # Define parser
        # class CodeBlockParser(BaseOutputParser):
        #     def parse(self, text: str) -> str:
        #         match = re.search(r'``````', text, re.DOTALL)
        #         return match.group(1).strip() if match else text

        class CodeBlockParser(BaseOutputParser):
            def parse(self,content:str):
                pattern = re.compile(r'```(?:python)?(.*?)```', re.DOTALL)
                match = pattern.search(content)
                if match:
                    return match.group(1).strip()
                return "no code"

        # Build chains
        self.correction_chain = self.correction_feedback_prompt | self.llm.with_structured_output(CorrectionFeedback)
        self.revision_chain = self.revision_prompt | self.llm | CodeBlockParser()

    def _build_graph(self):
        # State definition
        class State(TypedDict):
            initial_code: str
            error: str
            constitutional_principles: List[ConstitutionalPrinciple]
            correction_feedback_and_revised_code: List[Tuple[str, str]]
            revised_code: str
            execution_result: str
            executed_code: str
            execution_error: str

        # Define nodes
        async def apply_feedback_and_revise(state: State):
            feedback_revisions = []
            current_code = state["initial_code"]

            #if execution status is success then no need to run it
            if state["execution_result"]=="SUCCESS":
                return {
                "correction_feedback_and_revised_code": feedback_revisions,
                "revised_code": state["initial_code"]
            }

            #if not then run the revision chain 
            for principle in state["constitutional_principles"]:
                feedback = await self.correction_chain.ainvoke({
                    "code": current_code,
                    "error": state["error"],
                    "correction_request": principle.critique_request
                })
                if feedback["correction_needed"]:
                    revised = await self.revision_chain.ainvoke({
                        "code": current_code,
                        "error": state["error"],
                        "feedback": feedback["correction"],
                        "correction_request": principle.revision_request
                    })
                    current_code = revised
                    feedback_revisions.append((feedback["correction"], revised))
                else:
                    feedback_revisions.append((feedback["correction"], current_code))
            return {
                "correction_feedback_and_revised_code": feedback_revisions,
                "revised_code": current_code
            }

        async def execute_revised_code(state: State):
            revised_code = state["revised_code"]
            try:
                print("="*15+"Exicuting revised code"+"="*15)
                exec(revised_code, {})
                return {
                    "execution_result": "SUCCESS",
                    "executed_code": revised_code,
                    "execution_error": ""
                }
            except Exception as e:
                return {
                    "execution_result": "ERROR",
                    "executed_code": revised_code,
                    "execution_error": str(e)
                }

        # Build graph

        #adding nodes
        self.graph = StateGraph(State)
        self.graph.add_node("apply_feedback_and_revise", apply_feedback_and_revise)
        self.graph.add_node("execute_revised_code", execute_revised_code)

        #adding edges
        self.graph.add_edge(START, "apply_feedback_and_revise")
        self.graph.add_edge("apply_feedback_and_revise", "execute_revised_code")
        self.graph.add_edge("execute_revised_code", END)
        self.app = self.graph.compile()

    async def correct_code(
        self,
        initial_code: str,
        error: str,
        principles: List[ConstitutionalPrinciple]
    ) -> str:
        final_code = ""
        async for step in self.app.astream({
            "initial_code": initial_code,
            "error": error,
            "constitutional_principles": principles,
            "execution_result":"ERROR"
        }, stream_mode="values"):
            if step.get("execution_result"):
                final_code = step.get("revised_code", "")
        return final_code
