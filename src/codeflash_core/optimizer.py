from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from typing import TYPE_CHECKING

from codeflash_core.config import HIGH_EFFORT_TOP_N, EffortLevel
from codeflash_core.strategy import DefaultStrategy
from codeflash_core.strategy_utils import OptimizationRuntime
from codeflash_core.ui import console, paneled_text, progress_bar
from codeflash_core.ui import logger as ui_logger

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.config import CoreConfig
    from codeflash_core.models import FunctionToOptimize, OptimizationResult
    from codeflash_core.protocols import LanguagePlugin
    from codeflash_core.strategy_utils import OptimizationStrategy

logger = logging.getLogger(__name__)


class Optimizer:
    """Core optimization orchestrator.

    Drives the discover -> index -> rank -> per-function optimization loop.
    Delegates the actual optimization pipeline for each function to the active
    OptimizationStrategy (defaults to DefaultStrategy).
    """

    def __init__(
        self, config: CoreConfig, plugin: LanguagePlugin, strategy: OptimizationStrategy | None = None
    ) -> None:
        self.config = config
        self.plugin = plugin
        self.test_config = config.resolve_test_config()
        # Resolve the plugin's output comparator once (or None -> fallback to ==)
        self.output_comparator = getattr(plugin, "compare_outputs", None)
        self.cancel_event = threading.Event()
        # Share cancel event with plugin so it can abort long-running subprocess/HTTP calls
        if hasattr(plugin, "cancel_event"):
            object.__setattr__(plugin, "cancel_event", self.cancel_event)
        self.strategy = strategy or DefaultStrategy()

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self, files: list[Path], function_filter: str | None = None) -> list[OptimizationResult]:
        """Run the optimization pipeline on the given files.

        Returns a list of successful optimization results.
        """
        # Pre-run cleanup of leftover files from previous runs
        self.plugin_cleanup()
        self.cleanup_leftover_trace_files()

        with progress_bar("Discovering functions...", transient=True):
            functions = self.plugin.discover_functions(files)

        if function_filter:
            functions = [f for f in functions if function_filter in (f.function_name, f.qualified_name)]

        if not functions:
            ui_logger.info("No optimizable functions found.")
            return []

        ui_logger.info("Found %d functions to optimize.", len(functions))

        # Pre-index source files for dependency analysis (call graph)
        source_files = list({f.file_path for f in functions})

        def on_index_progress(result: object) -> None:
            pass

        with progress_bar("Building call graph...", transient=True):
            self.plugin.build_index(source_files, on_progress=on_index_progress)

        # Rank functions by impact (e.g. dependency count)
        functions = self.plugin.rank_functions(functions)

        results: list[OptimizationResult] = []
        skipped = 0
        cancelled = False

        try:
            for i, function in enumerate(functions):
                if self.is_cancelled():
                    cancelled = True
                    break

                console.rule(f"[bold][{i + 1}/{len(functions)}] {function.qualified_name}[/bold]")

                # Escalate top-N functions to HIGH effort when running at MEDIUM
                original_effort = self.config.effort
                if i < HIGH_EFFORT_TOP_N and self.config.effort == EffortLevel.MEDIUM.value:
                    self.config.effort = EffortLevel.HIGH.value

                result = self.optimize_function(function)
                self.config.effort = original_effort

                if result is not None:
                    results.append(result)
                    ui_logger.info("Optimized %s — %.2fx speedup", function.qualified_name, result.speedup)
                else:
                    skipped += 1
                    ui_logger.info("No improvement found for %s", function.qualified_name)
        except KeyboardInterrupt:
            ui_logger.warning("Keyboard interrupt received. Cleaning up…")
            cancelled = True
            self.cancel_event.set()
        finally:
            self.plugin_cleanup()
            self.cleanup_leftover_trace_files()

        console.rule()
        paneled_text(f"{len(functions)} analyzed, {len(results)} optimized, {skipped} skipped", title="Summary")
        return results

    def optimize_function(self, function: FunctionToOptimize) -> OptimizationResult | None:
        """Attempt to optimize a single function. Delegates to the active strategy."""
        if self.is_cancelled():
            return None

        runtime = OptimizationRuntime(
            plugin=self.plugin,
            config=self.config,
            test_config=self.test_config,
            cancel_event=self.cancel_event,
            output_comparator=self.output_comparator,
            trace_id=str(uuid.uuid4()),
        )
        return self.strategy.optimize_function(function, runtime)

    # -- Cleanup helpers (shared across all strategies) -------------------------

    def cleanup_leftover_trace_files(self) -> None:
        """Remove leftover .trace files from previous runs."""
        tests_root = self.test_config.tests_root
        if not tests_root.exists():
            return
        leftover = list(tests_root.glob("*.trace"))
        if leftover:
            logger.debug("Cleaning up %d leftover trace file(s)", len(leftover))
            for p in leftover:
                with contextlib.suppress(OSError):
                    p.unlink(missing_ok=True)

    def plugin_cleanup(self) -> None:
        """Delegate cleanup of language-specific leftover files to the plugin."""
        if hasattr(self.plugin, "cleanup_run"):
            try:
                self.plugin.cleanup_run(self.test_config.tests_root)
            except Exception:
                logger.debug("Plugin cleanup_run failed", exc_info=True)
