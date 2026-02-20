from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from codeflash.languages.python.static_analysis._ast import get_module_full_name, parse_imports


class ImportedInternalModuleAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    full_name: str
    file_path: Path

    @field_validator("name")
    @classmethod
    def name_is_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            msg = "must be an identifier"
            raise ValueError(msg)
        return v

    @field_validator("full_name")
    @classmethod
    def full_name_is_dotted_identifier(cls, v: str) -> str:
        if any(not s or not s.isidentifier() for s in v.split(".")):
            msg = "must be a dotted identifier"
            raise ValueError(msg)
        return v

    @field_validator("file_path")
    @classmethod
    def file_path_exists(cls, v: Path | None) -> Path | None:
        if v and not v.exists():
            msg = "must be an existing path"
            raise ValueError(msg)
        return v


def is_internal_module(module_name: str, project_root: Path) -> bool:
    module_path = module_name.replace(".", "/")
    possible_paths = [project_root / f"{module_path}.py", project_root / module_path / "__init__.py"]
    return any(path.exists() for path in possible_paths)


def get_module_file_path(module_name: str, project_root: Path) -> Path | None:
    module_path = module_name.replace(".", "/")
    possible_paths = [project_root / f"{module_path}.py", project_root / module_path / "__init__.py"]
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    return None


def analyze_imported_modules(
    code_str: str, module_file_path: Path, project_root: Path
) -> list[ImportedInternalModuleAnalysis]:
    """Statically finds and analyzes all imported internal modules."""
    module_rel_path = module_file_path.relative_to(project_root).with_suffix("")
    current_module = ".".join(module_rel_path.parts)
    imports = parse_imports(code_str)
    module_names: set[str] = set()
    for node in imports:
        module_names.update(get_module_full_name(node, current_module))
    internal_modules = {module_name for module_name in module_names if is_internal_module(module_name, project_root)}
    return [
        ImportedInternalModuleAnalysis(name=str(mod_name).split(".")[-1], full_name=mod_name, file_path=file_path)
        for mod_name in internal_modules
        if (file_path := get_module_file_path(mod_name, project_root)) is not None
    ]
