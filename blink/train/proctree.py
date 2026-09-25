"""Process trees on Windows: ending a trainer with all its children, and a kill-on-close job so a
supervisor's children die with it. Split out of blink.train.supervise, which uses both."""

import sys

import psutil


def kill_tree(pid: int, timeout_s: float) -> None:
    """Terminate a process and all its descendants (Windows has no process groups to signal)."""
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(alive, timeout=timeout_s)


_JOB_HANDLE: list[int] = []  # the kill-on-close job this process sits in; its handle is never closed
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def _job_limit_structure():
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6)] + [
            (name, ctypes.c_size_t)
            for name in ("ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed")
        ]

    return Extended


def bind_children_to_this_process() -> str:
    """Windows: put this process in a kill-on-close job, so every child it starts dies with it.

    A supervisor that is killed must never leave its trainer running with no stop rule watching.
    Children inherit the job; the only handle to it is held here, so when this process exits for any
    reason Windows closes the handle and terminates the trainer too. Returns what happened.
    """
    if _JOB_HANDLE:
        return "bound: already in a kill-on-close job"
    if sys.platform != "win32":
        return "not bound: kill-on-close jobs are a Windows feature"
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel32.CreateJobObjectW(None, None)
    limits = _job_limit_structure()()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    size = ctypes.sizeof(limits)
    configured = job and kernel32.SetInformationJobObject(
        ctypes.c_void_p(job), JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), size
    )
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    if not configured or not kernel32.AssignProcessToJobObject(
        ctypes.c_void_p(job), ctypes.c_void_p(kernel32.GetCurrentProcess())
    ):
        return f"not bound: Windows error {ctypes.get_last_error()}"
    _JOB_HANDLE.append(job)
    return "bound: children die with this supervisor"
