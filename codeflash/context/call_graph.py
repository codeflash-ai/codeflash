from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_qualified_name, path_belongs_to_site_packages
from codeflash.models.models import FunctionSource

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jedi.api.classes import Name


@dataclass(frozen=True, slots=True)
class IndexResult:
    file_path: Path
    cached: bool
    num_edges: int
    edges: tuple[tuple[str, str, bool], ...]  # (caller_qn, callee_name, is_cross_file)
    cross_file_edges: int
    error: bool


# ---------------------------------------------------------------------------
# Module-level helpers (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

_PARALLEL_THRESHOLD = 8

# Per-worker state, initialised by _init_index_worker in child processes
_worker_jedi_project: object | None = None
_worker_project_root_str: str | None = None


def _init_index_worker(project_root: str) -> None:
    import jedi

    global _worker_jedi_project, _worker_project_root_str
    _worker_jedi_project = jedi.Project(path=project_root)
    _worker_project_root_str = project_root


def _resolve_definitions(ref: Name) -> list[Name]:
    try:
        inferred = ref.infer()
        valid = [d for d in inferred if d.type in ("function", "class")]
        if valid:
            return valid
    except Exception:
        pass

    try:
        return ref.goto(follow_imports=True, follow_builtin_imports=False)
    except Exception:
        return []


def _is_valid_definition(definition: Name, caller_qualified_name: str, project_root_str: str) -> bool:
    definition_path = definition.module_path
    if definition_path is None:
        return False

    if not str(definition_path).startswith(project_root_str + os.sep):
        return False

    if path_belongs_to_site_packages(definition_path):
        return False

    if not definition.full_name or not definition.full_name.startswith(definition.module_name):
        return False

    if definition.type not in ("function", "class"):
        return False

    try:
        def_qn = get_qualified_name(definition.module_name, definition.full_name)
        if def_qn == caller_qualified_name:
            return False
    except ValueError:
        return False

    try:
        from codeflash.optimization.function_context import belongs_to_function_qualified

        if belongs_to_function_qualified(definition, caller_qualified_name):
            return False
    except Exception:
        pass

    return True


def _get_enclosing_function_qn(ref: Name) -> str | None:
    try:
        parent = ref.parent()
        if parent is None or parent.type != "function":
            return None
        if not parent.full_name or not parent.full_name.startswith(parent.module_name):
            return None
        return get_qualified_name(parent.module_name, parent.full_name)
    except (ValueError, AttributeError):
        return None


def _analyze_file(file_path: Path, jedi_project: object, project_root_str: str) -> tuple[set[tuple[str, ...]], bool]:
    """Pure Jedi analysis — no DB access. Returns (edges, had_error)."""
    import jedi

    resolved = str(file_path.resolve())

    try:
        script = jedi.Script(path=file_path, project=jedi_project)
        refs = script.get_names(all_scopes=True, definitions=False, references=True)
    except Exception:
        return set(), True

    edges: set[tuple[str, str, str, str, str, str, str, str]] = set()

    for ref in refs:
        try:
            caller_qn = _get_enclosing_function_qn(ref)
            if caller_qn is None:
                continue

            definitions = _resolve_definitions(ref)
            if not definitions:
                continue

            definition = definitions[0]
            definition_path = definition.module_path
            if definition_path is None:
                continue

            if not _is_valid_definition(definition, caller_qn, project_root_str):
                continue

            # Extract common edge components
            edge_base = (resolved, caller_qn, str(definition_path))

            if definition.type == "function":
                callee_qn = get_qualified_name(definition.module_name, definition.full_name)
                if len(callee_qn.split(".")) > 2:
                    continue
                edges.add(
                    (
                        *edge_base,
                        callee_qn,
                        definition.full_name,
                        definition.name,
                        definition.type,
                        definition.get_line_code(),
                    )
                )
            elif definition.type == "class":
                init_qn = get_qualified_name(definition.module_name, f"{definition.full_name}.__init__")
                if len(init_qn.split(".")) > 2:
                    continue
                edges.add(
                    (
                        *edge_base,
                        init_qn,
                        f"{definition.full_name}.__init__",
                        "__init__",
                        definition.type,
                        definition.get_line_code(),
                    )
                )
        except Exception:
            continue

    return edges, False


def _index_file_worker(args: tuple[str, str]) -> tuple[str, str, set[tuple[str, ...]], bool]:
    """Worker entry point for ProcessPoolExecutor."""
    file_path_str, file_hash = args
    edges, had_error = _analyze_file(Path(file_path_str), _worker_jedi_project, _worker_project_root_str)
    return file_path_str, file_hash, edges, had_error


# ---------------------------------------------------------------------------


class CallGraph:
    SCHEMA_VERSION = 1

    def __init__(self, project_root: Path, db_path: Path | None = None) -> None:
        import jedi

        self.project_root = project_root.resolve()
        self.project_root_str = str(self.project_root)
        self.jedi_project = jedi.Project(path=self.project_root)

        if db_path is None:
            from codeflash.code_utils.compat import codeflash_cache_db

            db_path = codeflash_cache_db

        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.indexed_file_hashes: dict[str, str] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS cg_schema_version (version INTEGER PRIMARY KEY)")
        row = cur.execute("SELECT version FROM cg_schema_version LIMIT 1").fetchone()
        if row is None:
            cur.execute("INSERT INTO cg_schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,))
        elif row[0] != self.SCHEMA_VERSION:
            # Schema mismatch — drop all cg_ tables and recreate
            cur.execute("DROP TABLE IF EXISTS cg_call_edges")
            cur.execute("DROP TABLE IF EXISTS cg_indexed_files")
            cur.execute("DELETE FROM cg_schema_version")
            cur.execute("INSERT INTO cg_schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,))

        cur.execute(
            """CREATE TABLE IF NOT EXISTS cg_indexed_files (
                project_root TEXT NOT NULL,
                file_path    TEXT NOT NULL,
                file_hash    TEXT NOT NULL,
                PRIMARY KEY (project_root, file_path)
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS cg_call_edges (
                project_root                TEXT NOT NULL,
                caller_file                 TEXT NOT NULL,
                caller_qualified_name       TEXT NOT NULL,
                callee_file                 TEXT NOT NULL,
                callee_qualified_name       TEXT NOT NULL,
                callee_fully_qualified_name TEXT NOT NULL,
                callee_only_function_name   TEXT NOT NULL,
                callee_definition_type      TEXT NOT NULL,
                callee_source_line          TEXT NOT NULL,
                PRIMARY KEY (project_root, caller_file, caller_qualified_name,
                             callee_file, callee_qualified_name)
            )"""
        )
        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_cg_edges_caller
               ON cg_call_edges (project_root, caller_file, caller_qualified_name)"""
        )
        self.conn.commit()

    def get_callees(
        self, file_path_to_qualified_names: dict[Path, set[str]]
    ) -> tuple[dict[Path, set[FunctionSource]], list[FunctionSource]]:
        file_path_to_function_source: dict[Path, set[FunctionSource]] = defaultdict(set)
        function_source_list: list[FunctionSource] = []

        # Build list of all caller keys
        all_caller_keys: list[tuple[str, str]] = []
        for file_path, qualified_names in file_path_to_qualified_names.items():
            self.ensure_file_indexed(file_path)
            resolved = str(file_path.resolve())
            all_caller_keys.extend((resolved, qn) for qn in qualified_names)

        if not all_caller_keys:
            return file_path_to_function_source, function_source_list

        # Query all callees
        cur = self.conn.cursor()
        for caller_file, caller_qn in all_caller_keys:
            rows = cur.execute(
                """SELECT callee_file, callee_qualified_name, callee_fully_qualified_name,
                          callee_only_function_name, callee_definition_type, callee_source_line
                   FROM cg_call_edges
                   WHERE project_root = ? AND caller_file = ? AND caller_qualified_name = ?""",
                (self.project_root_str, caller_file, caller_qn),
            ).fetchall()

            for callee_file, callee_qn, callee_fqn, callee_name, callee_type, callee_src in rows:
                callee_path = Path(callee_file)
                fs = FunctionSource(
                    file_path=callee_path,
                    qualified_name=callee_qn,
                    fully_qualified_name=callee_fqn,
                    only_function_name=callee_name,
                    source_code=callee_src,
                    definition_type=callee_type,
                )
                file_path_to_function_source[callee_path].add(fs)
                function_source_list.append(fs)

        return file_path_to_function_source, function_source_list

    def ensure_file_indexed(self, file_path: Path) -> IndexResult:
        resolved = str(file_path.resolve())

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return IndexResult(file_path=file_path, cached=False, num_edges=0, edges=(), cross_file_edges=0, error=True)

        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Check in-memory cache first
        if self.indexed_file_hashes.get(resolved) == file_hash:
            return IndexResult(file_path=file_path, cached=True, num_edges=0, edges=(), cross_file_edges=0, error=False)

        # Check DB for stored hash
        row = self.conn.execute(
            "SELECT file_hash FROM cg_indexed_files WHERE project_root = ? AND file_path = ?",
            (self.project_root_str, resolved),
        ).fetchone()

        if row and row[0] == file_hash:
            self.indexed_file_hashes[resolved] = file_hash
            return IndexResult(file_path=file_path, cached=True, num_edges=0, edges=(), cross_file_edges=0, error=False)

        return self.index_file(file_path, file_hash)

    def index_file(self, file_path: Path, file_hash: str) -> IndexResult:
        resolved = str(file_path.resolve())
        edges, had_error = _analyze_file(file_path, self.jedi_project, self.project_root_str)
        if had_error:
            logger.debug(f"CallGraph: failed to parse {file_path}")
        return self._persist_edges(file_path, resolved, file_hash, edges, had_error)

    def _persist_edges(
        self, file_path: Path, resolved: str, file_hash: str, edges: set[tuple[str, ...]], had_error: bool
    ) -> IndexResult:
        cur = self.conn.cursor()

        # Clear existing data for this file
        cur.execute(
            "DELETE FROM cg_call_edges WHERE project_root = ? AND caller_file = ?", (self.project_root_str, resolved)
        )
        cur.execute(
            "DELETE FROM cg_indexed_files WHERE project_root = ? AND file_path = ?", (self.project_root_str, resolved)
        )

        # Insert new edges if parsing succeeded
        if not had_error and edges:
            cur.executemany(
                """INSERT OR REPLACE INTO cg_call_edges
                   (project_root, caller_file, caller_qualified_name,
                    callee_file, callee_qualified_name, callee_fully_qualified_name,
                    callee_only_function_name, callee_definition_type, callee_source_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(self.project_root_str, *edge) for edge in edges],
            )

        # Record that this file has been indexed
        cur.execute(
            "INSERT OR REPLACE INTO cg_indexed_files (project_root, file_path, file_hash) VALUES (?, ?, ?)",
            (self.project_root_str, resolved, file_hash),
        )

        self.conn.commit()
        self.indexed_file_hashes[resolved] = file_hash

        # Build summary for return value
        edges_summary = tuple(
            (caller_qn, callee_name, caller_file != callee_file)
            for (caller_file, caller_qn, callee_file, _, _, callee_name, _, _) in edges
        )
        cross_file_count = sum(is_cross_file for _, _, is_cross_file in edges_summary)

        return IndexResult(
            file_path=file_path,
            cached=False,
            num_edges=len(edges),
            edges=edges_summary,
            cross_file_edges=cross_file_count,
            error=had_error,
        )

    def build_index(self, file_paths: Iterable[Path], on_progress: Callable[[IndexResult], None] | None = None) -> None:
        """Pre-index a batch of files, using multiprocessing for large uncached batches."""
        to_index: list[tuple[Path, str, str]] = []

        for file_path in file_paths:
            resolved = str(file_path.resolve())

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                self._report_progress(
                    on_progress,
                    IndexResult(
                        file_path=file_path, cached=False, num_edges=0, edges=(), cross_file_edges=0, error=True
                    ),
                )
                continue

            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Check if already cached (in-memory or DB)
            if self._is_file_cached(resolved, file_hash):
                self._report_progress(
                    on_progress,
                    IndexResult(
                        file_path=file_path, cached=True, num_edges=0, edges=(), cross_file_edges=0, error=False
                    ),
                )
                continue

            to_index.append((file_path, resolved, file_hash))

        if not to_index:
            return

        # Index uncached files
        if len(to_index) >= _PARALLEL_THRESHOLD:
            self._build_index_parallel(to_index, on_progress)
        else:
            for file_path, _resolved, file_hash in to_index:
                result = self.index_file(file_path, file_hash)
                self._report_progress(on_progress, result)

    def _is_file_cached(self, resolved: str, file_hash: str) -> bool:
        """Check if file is cached in memory or DB."""
        # Check in-memory cache
        if self.indexed_file_hashes.get(resolved) == file_hash:
            return True

        # Check DB cache
        row = self.conn.execute(
            "SELECT file_hash FROM cg_indexed_files WHERE project_root = ? AND file_path = ?",
            (self.project_root_str, resolved),
        ).fetchone()

        if row and row[0] == file_hash:
            self.indexed_file_hashes[resolved] = file_hash
            return True

        return False

    def _report_progress(self, on_progress: Callable[[IndexResult], None] | None, result: IndexResult) -> None:
        """Report progress if callback provided."""
        if on_progress is not None:
            on_progress(result)

    def _build_index_parallel(
        self, to_index: list[tuple[Path, str, str]], on_progress: Callable[[IndexResult], None] | None
    ) -> None:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        max_workers = min(os.cpu_count() or 1, len(to_index), 8)
        path_info: dict[str, tuple[Path, str]] = {resolved: (fp, fh) for fp, resolved, fh in to_index}
        worker_args = [(resolved, fh) for _fp, resolved, fh in to_index]

        logger.debug(f"CallGraph: indexing {len(to_index)} files across {max_workers} workers")

        try:
            with ProcessPoolExecutor(
                max_workers=max_workers, initializer=_init_index_worker, initargs=(self.project_root_str,)
            ) as executor:
                futures = {executor.submit(_index_file_worker, args): args[0] for args in worker_args}

                for future in as_completed(futures):
                    resolved = futures[future]
                    file_path, file_hash = path_info[resolved]

                    try:
                        _, _, edges, had_error = future.result()
                    except Exception:
                        logger.debug(f"CallGraph: worker failed for {file_path}")
                        self._persist_edges(file_path, resolved, file_hash, set(), had_error=True)
                        self._report_progress(
                            on_progress,
                            IndexResult(
                                file_path=file_path, cached=False, num_edges=0, edges=(), cross_file_edges=0, error=True
                            ),
                        )
                        continue

                    if had_error:
                        logger.debug(f"CallGraph: failed to parse {file_path}")

                    result = self._persist_edges(file_path, resolved, file_hash, edges, had_error)
                    self._report_progress(on_progress, result)

        except Exception:
            logger.debug("CallGraph: parallel indexing failed, falling back to sequential")
            self._fallback_sequential_index(to_index, on_progress)

    def _fallback_sequential_index(
        self, to_index: list[tuple[Path, str, str]], on_progress: Callable[[IndexResult], None] | None
    ) -> None:
        """Fallback to sequential indexing when parallel processing fails."""
        for file_path, resolved, file_hash in to_index:
            # Skip files already persisted before the failure
            if resolved in self.indexed_file_hashes:
                continue
            result = self.index_file(file_path, file_hash)
            self._report_progress(on_progress, result)

    def close(self) -> None:
        self.conn.close()
