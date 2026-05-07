from __future__ import annotations

import contextlib
import functools
import os
import shutil
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.git_utils import git_root_dir, mirror_path


class WorktreeSlot:
    __slots__ = ("_git_root", "index", "path")

    def __init__(self, path: Path, index: int, git_root: Path) -> None:
        self.path = path
        self.index = index
        self._git_root = git_root

    def mirror(self, original_path: Path) -> Path:
        return mirror_path(original_path, self._git_root, self.path)

    async def write_candidate(self, file_path: Path, code: str) -> None:
        mirrored = anyio.Path(self.mirror(file_path))
        await mirrored.parent.mkdir(parents=True, exist_ok=True)
        await mirrored.write_text(code, encoding="utf-8")

    async def restore_file(self, file_path: Path, original_code: str) -> None:
        mirrored = anyio.Path(self.mirror(file_path))
        await mirrored.write_text(original_code, encoding="utf-8")


class WorktreePool:
    def __init__(self, pool_size: int = 4, base_dir: Path | None = None) -> None:
        self._pool_size = pool_size
        self._git_root = git_root_dir()
        self._base_dir = base_dir or (self._git_root / ".codeflash_eval_worktrees")
        self._slots: list[WorktreeSlot] = []
        self._send: anyio.abc.ObjectSendStream[WorktreeSlot] | None = None
        self._receive: anyio.abc.ObjectReceiveStream[WorktreeSlot] | None = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await anyio.Path(self._base_dir).mkdir(parents=True, exist_ok=True)

        async with anyio.create_task_group() as tg:
            results: list[WorktreeSlot | None] = [None] * self._pool_size
            for i in range(self._pool_size):
                tg.start_soon(self._create_slot_task, i, results)

        self._slots = [s for s in results if s is not None]
        self._send, self._receive = anyio.create_memory_object_stream[WorktreeSlot](self._pool_size)
        for slot in self._slots:
            await self._send.send(slot)
        self._initialized = True
        logger.debug(f"WorktreePool initialized with {len(self._slots)} slots at {self._base_dir}")

    async def _create_slot_task(self, index: int, results: list[WorktreeSlot | None]) -> None:
        results[index] = await self._create_slot(index)

    async def _create_slot(self, index: int) -> WorktreeSlot:
        slot_dir = self._base_dir / f"slot-{index}"
        if slot_dir.exists():
            await anyio.to_thread.run_sync(functools.partial(shutil.rmtree, slot_dir, onerror=_handle_remove_readonly))

        result = await anyio.run_process(
            ["git", "-C", str(self._git_root), "worktree", "add", "--detach", str(slot_dir), "HEAD"], check=False
        )
        if result.returncode != 0:
            msg = f"Failed to create worktree slot {index}: {result.stderr.decode()}"
            raise RuntimeError(msg)

        pid_file = anyio.Path(slot_dir / ".codeflash_pool.pid")
        await pid_file.write_text(str(os.getpid()), encoding="utf-8")

        return WorktreeSlot(slot_dir, index, self._git_root)

    async def acquire(self) -> WorktreeSlot:
        assert self._receive is not None
        return await self._receive.receive()

    async def release(self, slot: WorktreeSlot) -> None:
        assert self._send is not None
        await self._send.send(slot)

    async def cleanup(self) -> None:
        async with anyio.create_task_group() as tg:
            for slot in self._slots:
                tg.start_soon(self._remove_slot_async, slot)
        self._slots.clear()
        self._initialized = False

        if self._base_dir.exists():
            with contextlib.suppress(Exception):
                await anyio.run_process(["git", "-C", str(self._git_root), "worktree", "prune"], check=False)
            with contextlib.suppress(OSError):
                self._base_dir.rmdir()

    async def _remove_slot_async(self, slot: WorktreeSlot) -> None:
        if slot.path.exists():
            await anyio.to_thread.run_sync(functools.partial(shutil.rmtree, slot.path, onerror=_handle_remove_readonly))

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.cleanup()


def _handle_remove_readonly(func: Callable[..., Any], path: str, exc_info: tuple[Any, ...]) -> None:
    if isinstance(exc_info[1], PermissionError):
        Path(path).chmod(stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(path)
    else:
        raise exc_info[1]
