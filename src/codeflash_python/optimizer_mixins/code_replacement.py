from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash_python.code_utils.formatter import format_code, sort_imports

if TYPE_CHECKING:
    from codeflash_python.models.models import CodeOptimizationContext, CodeStringsMarkdown, FunctionSource
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


class CodeReplacementMixin(_Base):
    @staticmethod
    def write_code_and_helpers(original_code: str, original_helper_code: dict[Path, str], path: Path) -> None:
        with path.open("w", encoding="utf8") as f:
            f.write(original_code)
        for module_abspath, helper_code in original_helper_code.items():
            with Path(module_abspath).open("w", encoding="utf8") as f:
                f.write(helper_code)

    def reformat_code_and_helpers(
        self,
        helper_functions: list[FunctionSource],
        path: Path,
        original_code: str,
        optimized_context: CodeStringsMarkdown,
    ) -> tuple[str, dict[Path, str]]:
        assert self.args is not None
        should_sort_imports = not self.args.disable_imports_sorting
        if should_sort_imports and sort_imports(code=original_code) != original_code:
            should_sort_imports = False

        optimized_code = ""
        if optimized_context is not None:
            file_to_code_context = optimized_context.file_to_path()
            optimized_code = file_to_code_context.get(str(path.resolve().relative_to(self.project_root)), "")

        new_code = format_code(
            self.args.formatter_cmds, path, optimized_code=optimized_code, check_diff=True, exit_on_failure=False
        )
        if should_sort_imports:
            new_code = sort_imports(new_code)

        new_helper_code: dict[Path, str] = {}
        for hp in helper_functions:
            module_abspath = hp.file_path
            hp_source_code = hp.source_code
            formatted_helper_code = format_code(
                self.args.formatter_cmds,
                module_abspath,
                optimized_code=hp_source_code,
                check_diff=True,
                exit_on_failure=False,
            )
            if should_sort_imports:
                formatted_helper_code = sort_imports(formatted_helper_code)
            new_helper_code[module_abspath] = formatted_helper_code

        return new_code, new_helper_code

    def group_functions_by_file(self, code_context: CodeOptimizationContext) -> dict[Path, set[str]]:
        functions_by_file: dict[Path, set[str]] = defaultdict(set)
        functions_by_file[self.function_to_optimize.file_path].add(self.function_to_optimize.qualified_name)
        for helper in code_context.helper_functions:
            if helper.definition_type in ("function", None):
                functions_by_file[helper.file_path].add(helper.qualified_name)
        return functions_by_file

    def revert_code_and_helpers(self, original_helper_code: dict[Path, str]) -> None:
        logger.info("Reverting code and helpers...")
        self.write_code_and_helpers(
            self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
        )
        self.cleanup_async_helper_file()
