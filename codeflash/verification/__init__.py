from pydantic.dataclasses import dataclass


@dataclass
class LLM:
    name: str
    max_tokens: int


@dataclass
class GPT_4_128k(LLM):
    name: str = "gpt-4-1106-preview"
    max_tokens: int = 128000


@dataclass
class GPT_4_32k(LLM):
    name: str = "gpt4-32k"
    max_tokens: int = 32768


@dataclass
class GPT_4(LLM):
    name: str = "gpt-4"
    max_tokens: int = 8192


@dataclass
class GPT_3_5_Turbo_16k(LLM):
    name: str = "gpt-3.5-turbo-16k"
    max_tokens: int = 16384


@dataclass
class GPT_3_5_Turbo(LLM):
    name: str = "gpt-3.5-turbo"
    max_tokens: int = 4096


EXPLAIN_MODEL: LLM = GPT_4_128k()
PLAN_MODEL = GPT_4_128k()
EXECUTE_MODEL = GPT_4_128k()
