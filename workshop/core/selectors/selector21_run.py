"""
Execution helpers for selector21.

Centralises the logic of dispatching tasks across serial/thread/process backends
so that selector21.py can focus on domain code.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, Tuple, Any, Iterator


Task = Sequence[Any]
WorkerFn = Callable[..., Any]
Result = Tuple[bool, Any]


def iter_task_results(
    tasks: Iterable[Task],
    worker: WorkerFn,
    *,
    backend: str = "thread",
    max_workers: int = 4,
) -> Iterator[Result]:
    """
    Execute worker(*task) for each task entry, respecting the selected backend.

    Yields tuples (success, payload) where 'payload' is either the worker return
    value (success=True) or the raised exception (success=False).
    """
    task_list = list(tasks)
    if not task_list:
        return

    backend = (backend or "serial").lower()

    if backend == "process":
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, *task) for task in task_list]
            for fut in as_completed(futures):
                try:
                    yield True, fut.result()
                except Exception as exc:
                    yield False, exc
    elif backend == "thread":
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, *task) for task in task_list]
            for fut in as_completed(futures):
                try:
                    yield True, fut.result()
                except Exception as exc:
                    yield False, exc
    else:
        for task in task_list:
            try:
                yield True, worker(*task)
            except Exception as exc:
                yield False, exc
