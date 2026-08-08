"""Execute untrusted candidate Octave code in a bounded local subprocess.

This backend is an alternative to the Prime Sandbox path in ``octave_rl.py``.
It exists because the managed Sandbox service is an independent failure and
billing domain: a run can lose its scoring path while GPU compute is healthy.
The local backend removes that dependency for development, CI, and pod-local
training, at the cost of a weaker containment boundary.

What does *not* change between backends is the property that actually protects
the reward signal. Both run the same input-only runner from
``harness.build_candidate_runner``, so hidden expected values and pass counters
never enter the interpreter that executes model output, and both score through
``harness.score_candidate_output`` in the trusted task process.

What *does* change is containment. A Prime Sandbox is a separate container on
separate hardware. This backend runs on the calling host and defends with a
scrubbed environment, a private working directory, an explicit resource
envelope, and a network namespace when the host allows one. Every record
reports the isolation it actually obtained rather than the isolation intended,
so a run can never be described as more contained than it was.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from asyncio import to_thread
from functools import lru_cache
from pathlib import Path
from typing import Any

from harness import build_candidate_runner, new_result_token, score_candidate_output

# The Sandbox backend bounds candidate execution with ``timeout=60`` on its
# ``execute_command`` call. Keep the wall-clock budget identical so a task that
# times out locally would also have timed out remotely.
EXECUTION_TIMEOUT_SECONDS = 60
# CPU time is bounded separately from wall clock so a spin loop cannot consume
# a core for the full timeout window.
CPU_SECONDS = 60
# The Sandbox requests 2 GB. Octave's BLAS reserves a large virtual mapping at
# startup, and RLIMIT_AS counts reservations, so a literal 2 GB cap fails on
# well-behaved code. This bounds runaway allocation without that false positive.
ADDRESS_SPACE_BYTES = 6 * 1024**3
OUTPUT_FILE_BYTES = 32 * 1024**2
MAX_PROCESSES = 64
# Only the terminal marker line is load-bearing, but keep enough context for
# the retry diagnostic while refusing to buffer an unbounded print loop.
MAX_CAPTURED_BYTES = 256 * 1024

OCTAVE_BIN_ENV = "OCTAVE_RL_OCTAVE_BIN"
OCTAVE_ROOTFS_ENV = "OCTAVE_RL_OCTAVE_ROOTFS"
ALLOW_UNISOLATED_ENV = "OCTAVE_RL_ALLOW_UNISOLATED_LOCAL"

# Preferred inside a pinned rootfs: the CLI build has no GUI dependencies.
ROOTFS_OCTAVE_CANDIDATES = (
    "/usr/local/bin/octave-cli",
    "/usr/bin/octave-cli",
    "/usr/local/bin/octave",
    "/usr/bin/octave",
)

FN_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


class LocalExecutionUnavailable(RuntimeError):
    """The local backend cannot run under the conditions it was asked to meet."""


def octave_rootfs() -> Path | None:
    """Return the unpacked pinned-image root, when one is configured.

    Pointing this at a rootfs extracted from ``gnuoctave/octave:10.2.0`` is the
    difference between scoring on *an* Octave and scoring on *the* Octave the
    reference pool was validated against. ``scripts/fetch_pinned_octave.py``
    produces one without needing a Docker daemon.
    """
    configured = os.environ.get(OCTAVE_ROOTFS_ENV)
    if not configured:
        return None
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise LocalExecutionUnavailable(
            f"{OCTAVE_ROOTFS_ENV} points at {root}, which is not a directory."
        )
    return root


def octave_binary(rootfs: Path | None = None) -> str:
    """Resolve the interpreter, inside the pinned rootfs when one is given.

    With a rootfs the returned path is the one the *chrooted* process will see,
    not a path on the calling host.
    """
    override = os.environ.get(OCTAVE_BIN_ENV)
    if rootfs is not None:
        for inner in (override,) if override else ROOTFS_OCTAVE_CANDIDATES:
            if (rootfs / inner.lstrip("/")).exists():
                return inner
        raise LocalExecutionUnavailable(
            f"no Octave interpreter found inside {rootfs}. Looked for "
            f"{', '.join(ROOTFS_OCTAVE_CANDIDATES)}."
        )
    candidate = override or "octave"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise LocalExecutionUnavailable(
            f"local executor needs an Octave interpreter; {candidate!r} is not on PATH. "
            f"Install GNU Octave, set {OCTAVE_BIN_ENV} to its absolute path, or set "
            f"{OCTAVE_ROOTFS_ENV} to an unpacked pinned image."
        )
    return resolved


def _host_tool(name: str) -> str:
    """Resolve a helper binary against the *parent's* PATH.

    The child runs with a scrubbed PATH that deliberately omits ``/usr/sbin``,
    where ``chroot`` lives on Debian-family images. Since ``execvpe`` resolves
    argv[0] against the environment being handed to the child, an unqualified
    name here fails to exec at all -- which looks exactly like every candidate
    producing empty output.
    """
    resolved = shutil.which(name)
    if resolved is not None:
        return resolved
    # A caller that already scrubbed its own PATH -- a training launcher, a
    # systemd unit -- may not list the sbin directories where chroot lives, so
    # check them directly before giving up on an otherwise capable host.
    for directory in ("/usr/sbin", "/sbin", "/usr/local/sbin"):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise LocalExecutionUnavailable(
        f"local executor needs {name!r}; not on PATH and not in /usr/sbin, /sbin, "
        "or /usr/local/sbin"
    )


@lru_cache(maxsize=1)
def network_isolation_prefix() -> tuple[str, ...]:
    """Return the argv prefix that denies the child a network, if one works.

    Probed rather than assumed: unprivileged user namespaces are disabled on
    some hosts and ``CAP_SYS_ADMIN`` is absent in some containers, and either
    would make an ``unshare`` prefix fail at exec time instead of quietly
    degrading. An empty result means no namespace was obtained.
    """
    unshare = shutil.which("unshare")
    if unshare is None:
        return ()
    for prefix in (
        (unshare, "--net", "--map-root-user", "--"),
        (unshare, "--net", "--"),
    ):
        try:
            probe = subprocess.run(
                [*prefix, "/bin/true"],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return prefix
    return ()


def _unisolated_is_allowed() -> bool:
    return os.environ.get(ALLOW_UNISOLATED_ENV, "").strip().lower() in {"1", "true", "yes"}


def _child_environment(inner_workdir: str) -> dict[str, str]:
    """Build the child environment from scratch.

    Constructed rather than filtered: an allow-list cannot be outrun by a new
    credential variable appearing in the parent process. Nothing from the
    caller's environment reaches candidate code, including ``PRIME_API_KEY``,
    Hugging Face tokens, and cloud provider credentials.
    """
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": inner_workdir,
        "TMPDIR": inner_workdir,
        "LC_ALL": "C",
        "LANG": "C",
        "TERM": "dumb",
        # Inside a chroot the image's /dev is not a real devfs, so keep the
        # history file in the disposable working directory instead.
        "OCTAVE_HISTFILE": f"{inner_workdir}/.octave_hist",
    }


def _limit_prologue() -> str:
    """Shell prologue that applies the resource envelope before exec.

    ``preexec_fn`` would be the obvious mechanism, but this runs inside
    ``asyncio.to_thread``: forking from a multithreaded process and then calling
    non-async-signal-safe code is a deadlock waiting to happen. A shell
    prologue sets the same limits in the child with no fork-time callback.

    Each limit is individually best-effort. ``ulimit -v`` is unsupported on
    macOS, and one unavailable limit should degrade that single bound rather
    than refuse to run the candidate at all.
    """
    return "; ".join(
        (
            f"ulimit -t {CPU_SECONDS} 2>/dev/null || :",
            f"ulimit -v {ADDRESS_SPACE_BYTES // 1024} 2>/dev/null || :",
            f"ulimit -f {OUTPUT_FILE_BYTES // 512} 2>/dev/null || :",
            "ulimit -c 0 2>/dev/null || :",
            f"ulimit -u {MAX_PROCESSES} 2>/dev/null || :",
        )
    )


def _candidate_argv(
    *,
    rootfs: Path | None,
    inner_workdir: str,
    interpreter: str,
    script: str,
) -> list[str]:
    """Assemble the full command, innermost shell first.

    Without a rootfs this is ``sh -c 'ulimits; exec octave script'``. With one,
    the same shell runs *inside* ``chroot``, so the limits apply to the chrooted
    process and the interpreter is the pinned build rather than the host's.
    """
    inner = (
        f"{_limit_prologue()}; cd {shlex.quote(inner_workdir)} && "
        f"exec {shlex.quote(interpreter)} --no-gui --no-window-system --norc "
        f"--quiet {shlex.quote(script)}"
    )
    isolation = list(network_isolation_prefix())
    if rootfs is None:
        return [*isolation, "/bin/sh", "-c", inner]
    return [*isolation, _host_tool("chroot"), str(rootfs), "/bin/sh", "-c", inner]


def _run_bounded(
    argv: list[str], *, workdir: Path, inner_workdir: str
) -> tuple[str, int]:
    """Run one candidate to completion, killing its whole process group on timeout."""
    process = subprocess.Popen(
        argv,
        cwd=workdir,
        env=_child_environment(inner_workdir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        # Interleave rather than append: only the final line carries the result
        # transport, and a trailing warning on a separate stream would displace
        # it and turn a correct answer into an unparseable one.
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, _ = process.communicate(timeout=EXECUTION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        stdout, _ = process.communicate()
    output = (stdout or b"").decode("utf-8", errors="replace")[-MAX_CAPTURED_BYTES:]
    if timed_out:
        output += f"\nTIMEOUT candidate exceeded {EXECUTION_TIMEOUT_SECONDS}s\n"
        return output, -1
    return output, process.returncode


def _execute_local_sync(
    *,
    fn_name: str,
    source: str,
    runner: str,
    cases: list[dict[str, Any]],
    tolerance: float,
    result_token: str,
) -> dict[str, Any]:
    if FN_NAME_RE.match(fn_name) is None:
        raise LocalExecutionUnavailable(f"refusing to write a candidate file for {fn_name!r}")
    prefix = network_isolation_prefix()
    if not prefix and not _unisolated_is_allowed():
        raise LocalExecutionUnavailable(
            "local executor could not obtain a network namespace on this host. "
            f"Set {ALLOW_UNISOLATED_ENV}=1 to run candidates with host network "
            "access anyway; every record will report network_isolated = false."
        )
    rootfs = octave_rootfs()
    interpreter = octave_binary(rootfs)
    # With a rootfs the working directory has to live inside it, since the
    # chrooted process cannot see anything above the new root.
    parent = str(rootfs / "workdir") if rootfs is not None else None
    if parent is not None:
        Path(parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="octave-rl-local-", dir=parent) as raw:
        workdir = Path(raw)
        workdir.chmod(0o700)
        (workdir / f"{fn_name}.m").write_text(source)
        (workdir / "run_candidate.m").write_text(runner)
        inner_workdir = (
            f"/workdir/{workdir.name}" if rootfs is not None else str(workdir)
        )
        output, exit_code = _run_bounded(
            _candidate_argv(
                rootfs=rootfs,
                inner_workdir=inner_workdir,
                interpreter=interpreter,
                script="run_candidate.m",
            ),
            workdir=workdir,
            inner_workdir=inner_workdir,
        )
    record = score_candidate_output(
        output,
        cases=cases,
        tolerance=tolerance,
        result_token=result_token,
        exit_code=exit_code,
    )
    record["runtime"] = "local"
    record["network_isolated"] = bool(prefix)
    record["filesystem_isolated"] = rootfs is not None
    return record


async def execute_candidate_locally(task: Any, source: str) -> dict[str, Any]:
    """Execute one candidate on this host and return the shared record.

    The signature mirrors the Sandbox entry points so the two backends are
    interchangeable at the call site.
    """
    result_token = new_result_token()
    return await to_thread(
        _execute_local_sync,
        fn_name=task.fn_name,
        source=source,
        runner=build_candidate_runner(task.model_dump(), result_token=result_token),
        cases=task.cases,
        tolerance=task.tolerance,
        result_token=result_token,
    )


def runtime_description() -> dict[str, Any]:
    """Report the interpreter and isolation this host will actually use.

    Intended to be recorded alongside any number the local runtime produces:
    "scored locally" is not a claim anyone can check, whereas "Octave 10.2.0
    from the pinned rootfs, network namespace obtained" is.
    """
    rootfs = octave_rootfs()
    interpreter = octave_binary(rootfs)
    isolation = network_isolation_prefix()
    argv = (
        [*isolation, _host_tool("chroot"), str(rootfs), interpreter, "--version"]
        if rootfs is not None
        else [interpreter, "--version"]
    )
    try:
        probe = subprocess.run(
            argv, capture_output=True, text=True, timeout=60, check=False
        )
        version = next(
            (
                line.strip()
                for line in (probe.stdout or "").splitlines()
                if "version" in line.lower()
            ),
            "unknown",
        )
    except (OSError, subprocess.SubprocessError):
        version = "unknown"
    return {
        "octave": version,
        "interpreter": interpreter,
        "rootfs": str(rootfs) if rootfs is not None else None,
        "network_isolated": bool(isolation),
        "isolation_prefix": " ".join(isolation) or None,
        "filesystem_isolated": rootfs is not None,
    }


__all__ = [
    "LocalExecutionUnavailable",
    "execute_candidate_locally",
    "network_isolation_prefix",
    "octave_binary",
    "octave_rootfs",
    "runtime_description",
]
