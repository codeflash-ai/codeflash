from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_qualified_name, path_belongs_to_site_packages
from codeflash.models.models import FunctionSource

if TYPE_CHECKING:
    from jedi.api.classes import Name


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

        all_caller_keys: list[tuple[str, str]] = []
        for file_path, qualified_names in file_path_to_qualified_names.items():
            self.ensure_file_indexed(file_path)
            fp_str = str(file_path.resolve())
            for qn in qualified_names:
                all_caller_keys.append((fp_str, qn))

        if not all_caller_keys:
            return file_path_to_function_source, function_source_list

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

    def ensure_file_indexed(self, file_path: Path) -> None:
        resolved = str(file_path.resolve())
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        cached_hash = self.indexed_file_hashes.get(resolved)
        if cached_hash == file_hash:
            return

        # Check DB for stored hash
        row = self.conn.execute(
            "SELECT file_hash FROM cg_indexed_files WHERE project_root = ? AND file_path = ?",
            (self.project_root_str, resolved),
        ).fetchone()
        if row and row[0] == file_hash:
            self.indexed_file_hashes[resolved] = file_hash
            return

        self.index_file(file_path, file_hash)

    def index_file(self, file_path: Path, file_hash: str) -> None:
        import jedi

        resolved = str(file_path.resolve())

        # Delete stale data for this file
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM cg_call_edges WHERE project_root = ? AND caller_file = ?", (self.project_root_str, resolved)
        )
        cur.execute(
            "DELETE FROM cg_indexed_files WHERE project_root = ? AND file_path = ?", (self.project_root_str, resolved)
        )

        try:
            script = jedi.Script(path=file_path, project=self.jedi_project)
            refs = script.get_names(all_scopes=True, definitions=False, references=True)
        except Exception:
            logger.debug(f"CallGraph: failed to parse {file_path}")
            cur.execute(
                "INSERT OR REPLACE INTO cg_indexed_files (project_root, file_path, file_hash) VALUES (?, ?, ?)",
                (self.project_root_str, resolved, file_hash),
            )
            self.conn.commit()
            self.indexed_file_hashes[resolved] = file_hash
            return

        edges: set[tuple[str, str, str, str, str, str, str, str]] = set()

        for ref in refs:
            try:
                caller_qn = self._get_enclosing_function_qualified_name(ref)
                if caller_qn is None:
                    continue

                definitions = self._resolve_definitions(ref)
                if not definitions:
                    continue

                definition = definitions[0]
                definition_path = definition.module_path
                if definition_path is None:
                    continue

                if not self._is_valid_definition(definition, caller_qn):
                    continue

                if definition.type == "function":
                    callee_qn = get_qualified_name(definition.module_name, definition.full_name)
                    if len(callee_qn.split(".")) > 2:
                        continue
                    edges.add(
                        (
                            resolved,
                            caller_qn,
                            str(definition_path),
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
                            resolved,
                            caller_qn,
                            str(definition_path),
                            init_qn,
                            f"{definition.full_name}.__init__",
                            "__init__",
                            definition.type,
                            definition.get_line_code(),
                        )
                    )
            except Exception:
                continue

        cur.executemany(
            """INSERT OR REPLACE INTO cg_call_edges
               (project_root, caller_file, caller_qualified_name,
                callee_file, callee_qualified_name, callee_fully_qualified_name,
                callee_only_function_name, callee_definition_type, callee_source_line)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(self.project_root_str, *edge) for edge in edges],
        )
        cur.execute(
            "INSERT OR REPLACE INTO cg_indexed_files (project_root, file_path, file_hash) VALUES (?, ?, ?)",
            (self.project_root_str, resolved, file_hash),
        )
        self.conn.commit()
        self.indexed_file_hashes[resolved] = file_hash

    def _resolve_definitions(self, ref: Name) -> list[Name]:
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

    def _is_valid_definition(self, definition: Name, caller_qualified_name: str) -> bool:
        definition_path = definition.module_path
        if definition_path is None:
            return False
        if not str(definition_path).startswith(self.project_root_str + os.sep):
            return False
        if path_belongs_to_site_packages(definition_path):
            return False
        if not definition.full_name or not definition.full_name.startswith(definition.module_name):
            return False
        if definition.type not in ("function", "class"):
            return False
        # No self-edges
        try:
            def_qn = get_qualified_name(definition.module_name, definition.full_name)
            if def_qn == caller_qualified_name:
                return False
        except ValueError:
            return False
        # Not an inner function of the caller
        try:
            from codeflash.optimization.function_context import belongs_to_function_qualified

            if belongs_to_function_qualified(definition, caller_qualified_name):
                return False
        except Exception:
            pass
        return True

    def _get_enclosing_function_qualified_name(self, ref: Name) -> str | None:
        try:
            parent = ref.parent()
            if parent is None or parent.type != "function":
                return None
            if not parent.full_name or not parent.full_name.startswith(parent.module_name):
                return None
            return get_qualified_name(parent.module_name, parent.full_name)
        except (ValueError, AttributeError):
            return None

    def close(self) -> None:
        self.conn.close()
