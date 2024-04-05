from pydantic import dataclasses


@dataclasses.dataclass
class FunctionModules:
    function_name: str
    file_name: str
    module_name: str
    class_name: str = None
