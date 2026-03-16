from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import encoded_tokens_len, get_run_tmp_file
from codeflash.code_utils.config_consts import (
    OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    READ_WRITABLE_LIMIT_ERROR,
    TESTGEN_CONTEXT_TOKEN_LIMIT,
    TESTGEN_LIMIT_ERROR,
    TOTAL_LOOPING_TIME_EFFECTIVE,
)
from codeflash.either import Failure, Success
from codeflash.languages.function_optimizer import FunctionOptimizer
from codeflash.models.models import (
    CodeOptimizationContext,
    CodeString,
    CodeStringsMarkdown,
    FunctionSource,
    TestingMode,
    TestResults,
)
from codeflash.verification.equivalence import compare_test_results

if TYPE_CHECKING:
    from codeflash.either import Result
    from codeflash.languages.base import CodeContext, HelperFunction
    from codeflash.models.models import CoverageData, GeneratedTestsList, OriginalCodeBaseline, TestDiff


class JavaFunctionOptimizer(FunctionOptimizer):
    def get_code_optimization_context(self) -> Result[CodeOptimizationContext, str]:
        from codeflash.languages import get_language_support
        from codeflash.languages.base import Language

        language = Language(self.function_to_optimize.language)
        lang_support = get_language_support(language)

        try:
            code_context = lang_support.extract_code_context(
                self.function_to_optimize, self.project_root, self.project_root
            )
            return Success(
                self._build_optimization_context(
                    code_context,
                    self.function_to_optimize.file_path,
                    self.function_to_optimize.language,
                    self.project_root,
                )
            )
        except ValueError as e:
            return Failure(str(e))

    @staticmethod
    def _build_optimization_context(
        code_context: CodeContext,
        file_path: Path,
        language: str,
        project_root: Path,
        optim_token_limit: int = OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
        testgen_token_limit: int = TESTGEN_CONTEXT_TOKEN_LIMIT,
    ) -> CodeOptimizationContext:
        imports_code = "\n".join(code_context.imports) if code_context.imports else ""

        try:
            target_relative_path = file_path.resolve().relative_to(project_root.resolve())
        except ValueError:
            target_relative_path = file_path

        helpers_by_file: dict[Path, list[HelperFunction]] = defaultdict(list)
        helper_function_sources = []

        for helper in code_context.helper_functions:
            helpers_by_file[helper.file_path].append(helper)
            helper_function_sources.append(
                FunctionSource(
                    file_path=helper.file_path,
                    qualified_name=helper.qualified_name,
                    fully_qualified_name=helper.qualified_name,
                    only_function_name=helper.name,
                    source_code=helper.source_code,
                )
            )

        target_file_code = code_context.target_code
        same_file_helpers = helpers_by_file.get(file_path, [])
        if same_file_helpers:
            helper_code = "\n\n".join(h.source_code for h in same_file_helpers)
            target_file_code = target_file_code + "\n\n" + helper_code

        if imports_code:
            target_file_code = imports_code + "\n\n" + target_file_code

        read_writable_code_strings = [
            CodeString(code=target_file_code, file_path=target_relative_path, language=language)
        ]

        for helper_file_path, file_helpers in helpers_by_file.items():
            if helper_file_path == file_path:
                continue
            try:
                helper_relative_path = helper_file_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                helper_relative_path = helper_file_path
            combined_helper_code = "\n\n".join(h.source_code for h in file_helpers)
            read_writable_code_strings.append(
                CodeString(code=combined_helper_code, file_path=helper_relative_path, language=language)
            )

        read_writable_code = CodeStringsMarkdown(code_strings=read_writable_code_strings, language=language)

        testgen_code_strings = read_writable_code_strings.copy()
        if code_context.imported_type_skeletons:
            testgen_code_strings.append(
                CodeString(code=code_context.imported_type_skeletons, file_path=None, language=language)
            )
        testgen_context = CodeStringsMarkdown(code_strings=testgen_code_strings, language=language)

        read_writable_tokens = encoded_tokens_len(read_writable_code.markdown)
        if read_writable_tokens > optim_token_limit:
            raise ValueError(READ_WRITABLE_LIMIT_ERROR)

        testgen_tokens = encoded_tokens_len(testgen_context.markdown)
        if testgen_tokens > testgen_token_limit:
            raise ValueError(TESTGEN_LIMIT_ERROR)

        code_hash = hashlib.sha256(read_writable_code.flat.encode("utf-8")).hexdigest()

        return CodeOptimizationContext(
            testgen_context=testgen_context,
            read_writable_code=read_writable_code,
            read_only_context_code=code_context.read_only_context,
            hashing_code_context=read_writable_code.flat,
            hashing_code_context_hash=code_hash,
            helper_functions=helper_function_sources,
            testgen_helper_fqns=[fs.fully_qualified_name for fs in helper_function_sources],
            preexisting_objects=set(),
        )

    def _get_java_sources_root(self) -> Path:
        """Get the Java sources root directory for test files.

        For Java projects, tests_root might include the package path
        (e.g., test/src/com/aerospike/test). We need to find the base directory
        that should contain the package directories, not the tests_root itself.

        This method looks for standard Java package prefixes (com, org, net, io, edu, gov)
        in the tests_root path and returns everything before that prefix.

        Returns:
            Path to the Java sources root directory.

        """
        tests_root = self.test_cfg.tests_root
        parts = tests_root.parts

        if tests_root.name == "src":
            return tests_root

        if len(parts) >= 3 and parts[-3:] == ("src", "test", "java"):
            return tests_root

        src_subdir = tests_root / "src"
        if src_subdir.exists() and src_subdir.is_dir():
            return src_subdir

        maven_test_dir = tests_root / "src" / "test" / "java"
        if maven_test_dir.exists() and maven_test_dir.is_dir():
            return maven_test_dir

        standard_package_prefixes = ("com", "org", "net", "io", "edu", "gov")
        for i, part in enumerate(parts):
            if part in standard_package_prefixes:
                if i > 0:
                    return Path(*parts[:i])

        for i, part in enumerate(parts):
            if part == "java" and i > 0:
                return Path(*parts[: i + 1])

        return tests_root

    def _fix_java_test_paths(
        self, behavior_source: str, perf_source: str, used_paths: set[Path], display_source: str = ""
    ) -> tuple[Path, Path, str, str, str]:
        """Fix Java test file paths to match package structure.

        Java requires test files to be in directories matching their package.
        This method extracts the package and class from the generated tests
        and returns correct paths. If the path would conflict with an already
        used path, it renames the class by adding an index suffix.

        Args:
            behavior_source: Source code of the behavior test.
            perf_source: Source code of the performance test.
            used_paths: Set of already used behavior file paths.
            display_source: Clean display version of the test (no instrumentation).

        Returns:
            Tuple of (behavior_path, perf_path, modified_behavior_source, modified_perf_source, modified_display_source)
            with correct package structure and unique class names.

        """
        package_match = re.search(r"^\s*package\s+([\w.]+)\s*;", behavior_source, re.MULTILINE)
        package_name = package_match.group(1) if package_match else ""

        # JPMS: If a test module-info.java exists, remap the package to the
        # test module namespace to avoid split-package errors.
        test_dir = self._get_java_sources_root()
        test_module_info = test_dir / "module-info.java"
        if package_name and test_module_info.exists():
            mi_content = test_module_info.read_text(encoding="utf-8")
            mi_match = re.search(r"module\s+([\w.]+)", mi_content)
            if mi_match:
                test_module_name = mi_match.group(1)
                main_dir = test_dir.parent.parent / "main" / "java"
                main_module_info = main_dir / "module-info.java"
                if main_module_info.exists():
                    main_content = main_module_info.read_text(encoding="utf-8")
                    main_match = re.search(r"module\s+([\w.]+)", main_content)
                    if main_match:
                        main_module_name = main_match.group(1)
                        if package_name.startswith(main_module_name):
                            suffix = package_name[len(main_module_name) :]
                            new_package = test_module_name + suffix
                            old_decl = f"package {package_name};"
                            new_decl = f"package {new_package};"
                            behavior_source = behavior_source.replace(old_decl, new_decl, 1)
                            perf_source = perf_source.replace(old_decl, new_decl, 1)
                            if display_source:
                                display_source = display_source.replace(old_decl, new_decl, 1)
                            package_name = new_package
                            logger.debug(f"[JPMS] Remapped package: {old_decl} -> {new_decl}")

        class_match = re.search(r"^(?:public\s+)?class\s+(\w+)", behavior_source, re.MULTILINE)
        behavior_class = class_match.group(1) if class_match else "GeneratedTest"

        perf_class_match = re.search(r"^(?:public\s+)?class\s+(\w+)", perf_source, re.MULTILINE)
        perf_class = perf_class_match.group(1) if perf_class_match else "GeneratedPerfTest"

        test_dir = self._get_java_sources_root()

        if package_name:
            package_path = package_name.replace(".", "/")
            behavior_path = test_dir / package_path / f"{behavior_class}.java"
            perf_path = test_dir / package_path / f"{perf_class}.java"
        else:
            package_path = ""
            behavior_path = test_dir / f"{behavior_class}.java"
            perf_path = test_dir / f"{perf_class}.java"

        modified_behavior_source = behavior_source
        modified_perf_source = perf_source
        modified_display_source = display_source
        if behavior_path in used_paths:
            index = 2
            while True:
                new_behavior_class = f"{behavior_class}_{index}"
                new_perf_class = f"{perf_class}_{index}"
                if package_path:
                    new_behavior_path = test_dir / package_path / f"{new_behavior_class}.java"
                    new_perf_path = test_dir / package_path / f"{new_perf_class}.java"
                else:
                    new_behavior_path = test_dir / f"{new_behavior_class}.java"
                    new_perf_path = test_dir / f"{new_perf_class}.java"
                if new_behavior_path not in used_paths:
                    behavior_path = new_behavior_path
                    perf_path = new_perf_path
                    # Rename ALL references to the class (not just declaration)
                    modified_behavior_source = re.sub(
                        rf"\b{re.escape(behavior_class)}\b", new_behavior_class, behavior_source
                    )
                    modified_perf_source = re.sub(rf"\b{re.escape(perf_class)}\b", new_perf_class, perf_source)
                    # Display source has the original (non-instrumented) class name
                    if display_source:
                        original_class = behavior_class.replace("__perfinstrumented", "")
                        new_original_class = f"{original_class}_{index}"
                        modified_display_source = re.sub(
                            rf"\b{re.escape(original_class)}\b", new_original_class, display_source
                        )
                    logger.debug(f"[JAVA] Renamed duplicate test class from {behavior_class} to {new_behavior_class}")
                    break
                index += 1

        behavior_path.parent.mkdir(parents=True, exist_ok=True)
        perf_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"[JAVA] Fixed paths: behavior={behavior_path}, perf={perf_path}")
        return behavior_path, perf_path, modified_behavior_source, modified_perf_source, modified_display_source

    def fixup_generated_tests(self, generated_tests: GeneratedTestsList) -> GeneratedTestsList:
        from codeflash.models.models import GeneratedTests, GeneratedTestsList

        used_paths: set[Path] = set()
        fixed_tests: list[GeneratedTests] = []
        for test in generated_tests.generated_tests:
            behavior_path, perf_path, behavior_source, perf_source, display_source = self._fix_java_test_paths(
                test.instrumented_behavior_test_source,
                test.instrumented_perf_test_source,
                used_paths,
                test.generated_original_test_source,
            )
            used_paths.add(behavior_path)
            fixed_tests.append(
                GeneratedTests(
                    generated_original_test_source=display_source,
                    instrumented_behavior_test_source=behavior_source,
                    instrumented_perf_test_source=perf_source,
                    behavior_file_path=behavior_path,
                    perf_file_path=perf_path,
                )
            )
        return GeneratedTestsList(generated_tests=fixed_tests)

    def compare_candidate_results(
        self,
        baseline_results: OriginalCodeBaseline,
        candidate_behavior_results: TestResults,
        optimization_candidate_index: int,
    ) -> tuple[bool, list[TestDiff]]:
        original_sqlite = get_run_tmp_file(Path("test_return_values_0.sqlite"))
        candidate_sqlite = get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite"))

        if len(baseline_results.behavior_test_results) == 0 or len(candidate_behavior_results) == 0:
            return False, []

        if original_sqlite.exists() and candidate_sqlite.exists():
            match, diffs = self.language_support.compare_test_results(
                original_sqlite,
                candidate_sqlite,
                project_root=self.project_root,
                project_classpath=self._get_project_classpath(),
            )
            candidate_sqlite.unlink(missing_ok=True)
        else:
            match, diffs = compare_test_results(baseline_results.behavior_test_results, candidate_behavior_results)
        return match, diffs

    def _get_project_classpath(self) -> str | None:
        """Get the project's full classpath from the build tool strategy.

        The classpath is cached by the strategy after the first test run,
        so this is a cheap dict lookup.
        """
        try:
            import os

            from codeflash.languages.java.build_tool_strategy import get_strategy

            strategy = get_strategy(self.project_root)
            return strategy.get_classpath(self.project_root, os.environ.copy(), None, timeout=60)
        except Exception:
            logger.debug("Could not get project classpath for Comparator", exc_info=True)
            return None

    def should_skip_sqlite_cleanup(self, testing_type: TestingMode, optimization_iteration: int) -> bool:
        return testing_type == TestingMode.BEHAVIOR or optimization_iteration == 0

    def parse_line_profile_test_results(
        self, line_profiler_output_file: Path | None
    ) -> tuple[TestResults | dict[str, Any], CoverageData | None]:
        if line_profiler_output_file is None or not line_profiler_output_file.exists():
            return TestResults(test_results=[]), None
        if hasattr(self.language_support, "parse_line_profile_results"):
            return self.language_support.parse_line_profile_results(line_profiler_output_file), None
        return TestResults(test_results=[]), None

    def line_profiler_step(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict[str, Any]:
        if not hasattr(self.language_support, "instrument_source_for_line_profiler"):
            logger.warning(f"Language support for {self.language_support.language} doesn't support line profiling")
            return {"timings": {}, "unit": 0, "str_out": ""}

        original_source = self.function_to_optimize.file_path.read_text(encoding="utf-8")
        try:
            line_profiler_output_path = get_run_tmp_file(Path("line_profiler_output.json"))

            success = self.language_support.instrument_source_for_line_profiler(
                func_info=self.function_to_optimize, line_profiler_output_file=line_profiler_output_path
            )
            if not success:
                return {"timings": {}, "unit": 0, "str_out": ""}

            test_env = self.get_test_env(
                codeflash_loop_index=0, codeflash_test_iteration=candidate_index, codeflash_tracer_disable=1
            )

            _test_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.LINE_PROFILE,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                enable_coverage=False,
                code_context=code_context,
                line_profiler_output_file=line_profiler_output_path,
            )

            return self.language_support.parse_line_profile_results(line_profiler_output_path)
        except Exception as e:
            logger.warning(f"Failed to run line profiling: {e}")
            return {"timings": {}, "unit": 0, "str_out": ""}
        finally:
            self.function_to_optimize.file_path.write_text(original_source, encoding="utf-8")

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        optimized_code: CodeStringsMarkdown,
        original_helper_code: dict[Path, str],
    ) -> bool:
        did_update = False
        for module_abspath, qualified_names in self.group_functions_by_file(code_context).items():
            did_update |= self.language_support.replace_function_definitions(
                function_names=list(qualified_names),
                optimized_code=optimized_code,
                module_abspath=module_abspath,
                project_root_path=self.project_root,
                function_to_optimize=self.function_to_optimize,
            )
        return did_update
