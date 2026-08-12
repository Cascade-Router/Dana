"""Windows Job Object sandbox for containing child processes.

Uses ``win32job`` (pywin32) when available, otherwise ``ctypes.windll.kernel32``.
On non-Windows or when Job APIs are unavailable, ``WindowsJob`` is a no-op
context manager so tests can import this module safely.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any

__all__ = [
    "JOB_APIS_AVAILABLE",
    "WindowsJob",
    "resume_suspended_process",
]

JOB_APIS_AVAILABLE = False
_BACKEND: str | None = None

if os.name == "nt":
    try:
        import win32api  # noqa: F401
        import win32con  # noqa: F401
        import win32job  # noqa: F401

        JOB_APIS_AVAILABLE = True
        _BACKEND = "pywin32"
    except ImportError:
        try:
            import ctypes  # noqa: F401

            JOB_APIS_AVAILABLE = True
            _BACKEND = "ctypes"
        except ImportError:
            JOB_APIS_AVAILABLE = False
            _BACKEND = None


# Win32 constants (ctypes path / shared)
_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_PROCESS_SUSPEND_RESUME = 0x0800
_THREAD_SUSPEND_RESUME = 0x0002
_TH32CS_SNAPTHREAD = 0x00000004


def _ctypes_kernel32() -> Any:
    import ctypes

    return ctypes.windll.kernel32


def _ctypes_set_kill_on_close(h_job: Any) -> None:
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    k32 = _ctypes_kernel32()
    ret_len = wintypes.DWORD(0)
    k32.QueryInformationJobObject(
        h_job,
        _JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
        ctypes.byref(ret_len),
    )
    info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(
        h_job,
        _JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        raise OSError("SetInformationJobObject failed")


class WindowsJob:
    """Context manager for a Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.

    Closing the job handle terminates any processes still assigned to it.
    On non-Windows / missing APIs this is a no-op so callers can keep one path.
    """

    def __init__(self) -> None:
        self._handle: Any = None
        self._active = False

    def __enter__(self) -> WindowsJob:
        if os.name != "nt" or not JOB_APIS_AVAILABLE:
            return self

        if _BACKEND == "pywin32":
            import win32job

            self._handle = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(
                self._handle, win32job.JobObjectExtendedLimitInformation
            )
            info["BasicLimitInformation"]["LimitFlags"] |= (
                win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            win32job.SetInformationJobObject(
                self._handle, win32job.JobObjectExtendedLimitInformation, info
            )
            self._active = True
            return self

        # ctypes fallback
        import ctypes
        from ctypes import wintypes

        k32 = _ctypes_kernel32()
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        h_job = k32.CreateJobObjectW(None, None)
        if not h_job:
            raise OSError("CreateJobObjectW failed")
        self._handle = h_job
        _ctypes_set_kill_on_close(h_job)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        handle = self._handle
        self._handle = None
        self._active = False
        if handle is None or os.name != "nt":
            return
        try:
            if _BACKEND == "pywin32":
                import win32api

                win32api.CloseHandle(handle)
            else:
                _ctypes_kernel32().CloseHandle(handle)
        except Exception:  # noqa: BLE001
            pass

    @property
    def active(self) -> bool:
        return self._active

    def assign_pid(self, pid: int) -> bool:
        """Open ``pid`` and assign it to this job. Returns True on success."""
        if not self._active or self._handle is None or pid is None:
            return False
        if _BACKEND == "pywin32":
            import win32api
            import win32con
            import win32job

            access = (
                win32con.PROCESS_SET_QUOTA
                | win32con.PROCESS_TERMINATE
                | getattr(win32con, "PROCESS_SUSPEND_RESUME", _PROCESS_SUSPEND_RESUME)
            )
            h_proc = win32api.OpenProcess(access, False, int(pid))
            try:
                win32job.AssignProcessToJobObject(self._handle, h_proc)
                return True
            finally:
                win32api.CloseHandle(h_proc)

        from ctypes import wintypes

        k32 = _ctypes_kernel32()
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        access = _PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_SUSPEND_RESUME
        h_proc = k32.OpenProcess(access, False, int(pid))
        if not h_proc:
            return False
        try:
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            return bool(k32.AssignProcessToJobObject(self._handle, h_proc))
        finally:
            k32.CloseHandle(h_proc)

    def assign_handle(self, process_handle: Any) -> bool:
        """Assign an open process handle to this job. Returns True on success."""
        if not self._active or self._handle is None or process_handle is None:
            return False
        if _BACKEND == "pywin32":
            import win32job

            win32job.AssignProcessToJobObject(self._handle, process_handle)
            return True

        from ctypes import wintypes

        k32 = _ctypes_kernel32()
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        return bool(
            k32.AssignProcessToJobObject(self._handle, wintypes.HANDLE(int(process_handle)))
        )


def resume_suspended_process(pid: int) -> bool:
    """Resume a process started with ``CREATE_SUSPENDED``.

    Prefers ``ctypes`` ``ResumeThread`` on the process's threads; falls back to
    ``psutil.Process.resume()`` when available.
    """
    if os.name != "nt" or pid is None:
        return False

    try:
        if _resume_threads_ctypes(int(pid)):
            return True
    except Exception:  # noqa: BLE001
        pass

    try:
        import psutil

        psutil.Process(int(pid)).resume()
        return True
    except Exception:  # noqa: BLE001
        return False


def _resume_threads_ctypes(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    k32 = _ctypes_kernel32()
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    k32.Thread32First.restype = wintypes.BOOL
    k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
    k32.Thread32Next.restype = wintypes.BOOL
    k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenThread.restype = wintypes.HANDLE
    k32.ResumeThread.argtypes = [wintypes.HANDLE]
    k32.ResumeThread.restype = wintypes.DWORD
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    snap = k32.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
    if not snap or int(snap) == -1:
        return False

    resumed = False
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(THREADENTRY32)
        more = k32.Thread32First(snap, ctypes.byref(entry))
        while more:
            if int(entry.th32OwnerProcessID) == int(pid):
                h_thread = k32.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if h_thread:
                    try:
                        # Resume until not suspended (CREATE_SUSPENDED => count 1).
                        while True:
                            prev = k32.ResumeThread(h_thread)
                            if prev == 0xFFFFFFFF or prev <= 1:
                                break
                        resumed = True
                    finally:
                        k32.CloseHandle(h_thread)
            more = k32.Thread32Next(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return resumed
