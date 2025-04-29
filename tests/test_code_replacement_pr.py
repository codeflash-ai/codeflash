from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

class Args:
    disable_imports_sorting = True
    formatter_cmds = ["disabled"]
def test_code_replacement_pr():
    optimized_code = """from typing import List, Optional

import requests
from inference.core.env import API_BASE_URL
from openai import OpenAI
from openai._types import NOT_GIVEN

# Create a global requests session to reuse connections
sess = requests.Session()

# Create a cache for OpenAI clients to avoid recreating them frequently
openai_clients = {}

def _execute_proxied_openai_request(
    roboflow_api_key: str,
    openai_api_key: str,
    prompt: List[dict],
    gpt_model_version: str,
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    \"\"\"Executes OpenAI request via Roboflow proxy.\"\"\"
    payload = {
        \"model\": gpt_model_version,
        \"messages\": prompt,
        \"max_tokens\": max_tokens,
        \"openai_api_key\": openai_api_key,
    }
    if temperature is not None:
        payload[\"temperature\"] = temperature

    try:
        endpoint = f\"{API_BASE_URL}/apiproxy/openai?api_key={roboflow_api_key}\"
        # Use global session for requests
        response = sess.post(endpoint, json=payload)
        response.raise_for_status()
        response_data = response.json()
        return response_data[\"choices\"][0][\"message\"][\"content\"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f\"Failed to connect to Roboflow proxy: {e}\") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f\"Invalid response structure from Roboflow proxy: {e} - Response: {response.text}\"
        ) from e


def _execute_openai_request(
    openai_api_key: str,
    prompt: List[dict],
    gpt_model_version: str,
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    \"\"\"Executes OpenAI request directly.\"\"\"
    temp_value = temperature if temperature is not None else NOT_GIVEN
    try:
        # Cache OpenAI client to avoid creating a new one each time
        if openai_api_key not in openai_clients:
            openai_clients[openai_api_key] = OpenAI(api_key=openai_api_key)
        client = openai_clients[openai_api_key]
        
        response = client.chat.completions.create(
            model=gpt_model_version,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temp_value,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f\"OpenAI API request failed: {e}\") from e


def execute_gpt_4v_request(
    roboflow_api_key: str,
    openai_api_key: str,
    prompt: List[dict],
    gpt_model_version: str,
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    if openai_api_key.startswith(\"rf_key:account\") or openai_api_key.startswith(
        \"rf_key:user:\"
    ):
        return _execute_proxied_openai_request(
            roboflow_api_key=roboflow_api_key,
            openai_api_key=openai_api_key,
            prompt=prompt,
            gpt_model_version=gpt_model_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        return _execute_openai_request(
            openai_api_key=openai_api_key,
            prompt=prompt,
            gpt_model_version=gpt_model_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
"""

    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/roboflow.py").resolve()
    original_code_str = (Path(__file__).parent.resolve() / "../code_to_optimize/roboflow_original.py").read_text(encoding="utf-8")
    code_path.write_text(original_code_str, encoding="utf-8")
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    func = FunctionToOptimize(function_name="execute_gpt_4v_request", parents=[], file_path=code_path)
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    code_context: CodeOptimizationContext = func_optimizer.get_code_optimization_context().unwrap()
    original_helper_code: dict[Path, str] = {}
    helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
    for helper_function_path in helper_function_paths:
        with helper_function_path.open(encoding="utf8") as f:
            helper_code = f.read()
            original_helper_code[helper_function_path] = helper_code
    func_optimizer.args = Args()
    func_optimizer.replace_function_and_helpers_with_optimized_code(
        code_context=code_context, optimized_code=optimized_code
    )
    new_code, new_helper_code = func_optimizer.reformat_code_and_helpers(
        code_context.helper_functions, func.file_path, func_optimizer.function_to_optimize_source_code
    )
    original_code_combined = original_helper_code.copy()
    original_code_combined[func.file_path] = func_optimizer.function_to_optimize_source_code
    new_code_combined = new_helper_code.copy()
    new_code_combined[func.file_path] = new_code
    final_output = code_path.read_text(encoding="utf-8")
    assert "openai_clients = {}" in final_output
    code_path.unlink(missing_ok=True)