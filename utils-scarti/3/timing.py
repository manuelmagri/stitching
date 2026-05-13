"""Logging dei tempi di esecuzione delle funzioni della pipeline.

Uso tipico:

    from utils.timing import timed, set_log_path

    set_log_path("output/timings_0_to_177.log")

    @timed
    def render_ortho(...):
        ...

Se `set_log_path` non è stata chiamata, il decoratore esegue la funzione
senza scrivere nulla — utile per import diretti e test.

Il file viene troncato a ogni nuova sessione (`set_log_path`) e poi scritto
in append. La scrittura è protetta da un lock, così è safe anche se in
futuro la pipeline diventasse multithread.
"""
import functools
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

_log_path: Optional[Path] = None
_lock = threading.Lock()


def set_log_path(path: Union[str, Path, None]) -> None:
    """Imposta il file di log e scrive l'header. `None` disabilita il logging."""
    global _log_path
    if path is None:
        _log_path = None
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(f"# Timings session — {datetime.now().isoformat(timespec='seconds')}\n")
    _log_path = p


def _format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds * 1e6:.1f}µs"


def _write_line(msg: str) -> None:
    if _log_path is None:
        return
    with _lock, _log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def timed(func):
    """Cronometra `func` e scrive una riga `[hh:mm:ss.mmm] name — duration` sul log."""
    qualname = f"{func.__module__}.{func.__qualname__}"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            _write_line(f"[{ts}] {qualname} — {_format_duration(elapsed)}")

    return wrapper
