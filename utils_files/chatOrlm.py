from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# from langchain.output_parsers.structured import StructuredOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import LLM
from pydantic import BaseModel, PrivateAttr # Use this if you've migrated to Pydantic v2
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from typing import Optional, List
from langchain_core.callbacks import CallbackManagerForLLMRun

class ChatOrlm(LLM):
    model_path: str
    max_new_tokens: int = 1000000

    _tokenizer: any = PrivateAttr()
    _model: any = PrivateAttr()
    _generator: any = PrivateAttr()
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.model_path = "./models/ORLM"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self._generator = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer)



    def _call(self,prompt: str,stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> str:
        response = self._generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=True)
        return response[0]["generated_text"][len(prompt):]

    # def _call(self, prompt: str, stop: list = None) -> str:
    #     response = self._generator(prompt, max_new_tokens=self.max_new_tokens, do_sample=True)
    #     return response[0]["generated_text"][len(prompt):]

    @property
    def _llm_type(self) -> str:
        return "local"
    def with_structured_output(self, schema: type[BaseModel]):
        parser = PydanticOutputParser(pydantic_object=schema)
        return RunnableLambda(lambda prompt: self.invoke(f"{prompt}\n{parser.get_format_instructions()}")) | parser
    
    # def with_structured_output(self, schema: type[BaseModel]):
    #     structurer = _get_structured_output_function(schema)
    #     return RunnableLambda(self.invoke) | StrOutputParser() | structurer