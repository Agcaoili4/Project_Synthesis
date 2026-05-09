"""Fire-and-forget task runner for work that must outlive a request."""

import asyncio
import logging
from collections.abc import Awaitable

log = logging.getLogger("synthesis.background")


class BackgroundTaskRunner:
    """Tracks fire-and-forget coroutines so shutdown can drain them.

    Used by ConverseUseCase to embed+store a finished turn-pair after the
    HTTP response has returned. Bounded by ``max_pending`` so a stuck
    embedder cannot grow the queue without bound.
    """

    def __init__(self, max_pending: int = 32) -> None:
        self._max_pending = max_pending
        self._tasks: set[asyncio.Task[None]] = set()

    def schedule(self, coro: Awaitable[None]) -> bool:
        if len(self._tasks) >= self._max_pending:
            log.warning(
                "background queue saturated (%d pending); dropping task",
                self._max_pending,
            )
            coro.close()  # type: ignore[attr-defined]
            return False
        task = asyncio.create_task(self._wrap(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    @staticmethod
    async def _wrap(coro: Awaitable[None]) -> None:
        try:
            await coro
        except Exception:
            log.exception("background task raised")

    async def drain(self, timeout: float = 5.0) -> None:
        if not self._tasks:
            return
        log.info("draining %d background task(s)", len(self._tasks))
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            log.warning(
                "background drain timed out after %.1fs (%d still pending)",
                timeout,
                len(self._tasks),
            )

    @property
    def pending(self) -> int:
        return len(self._tasks)
