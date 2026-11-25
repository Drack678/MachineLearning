"""Utilidades de logging legible y silencio de spam nativo ROCm/MIOpen para Workshop 5."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Iterator


def configure_rocm_env() -> None:
    """Configura variables de entorno de MIOpen antes de importar torch.

    Respeta variables definidas manualmente por el usuario y se omite si se pasa
    ``--verbose-rocm`` por línea de comandos.
    """
    if "--verbose-rocm" in sys.argv:
        return
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "1")
    os.environ.setdefault("MIOPEN_ENABLE_LOGGING", "0")
    os.environ.setdefault("MIOPEN_ENABLE_LOGGING_CMD", "0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


def format_duration(seconds: float) -> str:
    """Formatea una duración en segundos a texto compacto ASCII."""
    secs = max(0.0, float(seconds))
    if secs < 60.0:
        return f"{secs:.1f}s"
    total_s = int(round(secs))
    mins, rem_s = divmod(total_s, 60)
    if mins < 60:
        return f"{mins}m {rem_s:02d}s"
    hours, rem_m = divmod(mins, 60)
    return f"{hours}h {rem_m:02d}m"


def log(msg: str) -> None:
    """Imprime una línea con timestamp y flush inmediato."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@contextlib.contextmanager
def silence_native_rocm_logs(enabled: bool = True) -> Iterator[None]:
    """Redirige stdout/stderr nativos de C++ (MIOpen/CK) a devnull sin ocultar Python.

    Los ``print()`` y tracebacks de Python siguen saliendo normalmente por la terminal o pipe.
    """
    if not enabled:
        yield
        return

    try:
        out_fd = sys.stdout.fileno()
        err_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    try:
        saved_out = os.dup(out_fd)
        saved_err = os.dup(err_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return

    new_stdout = None
    new_stderr = None
    try:
        os.dup2(devnull_fd, out_fd)
        os.dup2(devnull_fd, err_fd)
        new_stdout = os.fdopen(
            saved_out,
            "w",
            encoding=getattr(orig_stdout, "encoding", None) or "utf-8",
            errors="replace",
            buffering=1,
        )
        new_stderr = os.fdopen(
            saved_err,
            "w",
            encoding=getattr(orig_stderr, "encoding", None) or "utf-8",
            errors="replace",
            buffering=1,
        )
        sys.stdout = new_stdout
        sys.stderr = new_stderr
        yield
    finally:
        if new_stdout is not None:
            new_stdout.flush()
        if new_stderr is not None:
            new_stderr.flush()
        os.dup2(saved_out, out_fd)
        os.dup2(saved_err, err_fd)
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if new_stdout is not None:
            new_stdout.close()
        else:
            os.close(saved_out)
        if new_stderr is not None:
            new_stderr.close()
        else:
            os.close(saved_err)
        os.close(devnull_fd)
