"""Steady conditions for timing a single process.

The laptop the November 2026 experiments ran on (AMD Ryzen AI 9 HX 370) has two kinds of cores, four fast Zen 5 and
eight slower Zen 5c, and Windows may move a long-running background process to a slower core or clock (power
throttling, EcoQoS). A search then takes up to twice as long for the same nodes, and the times of one run do not
compare. steady() pins the process to one logical CPU, raises its priority and opts out of power throttling; on
other systems it does nothing. Node counts never depend on it, only times and the depth reached under a clock.
"""
import os
import sys


def steady(cpu=None):
    """Pin this process to logical CPU `cpu` (default: $BENCH_CPU or 0). Returns a description for the results."""
    if sys.platform != "win32":
        return "not pinned (%s)" % sys.platform
    import ctypes
    from ctypes import wintypes

    cpu = int(os.environ.get("BENCH_CPU", "0")) if cpu is None else cpu
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    h = k.GetCurrentProcess()
    if not k.SetProcessAffinityMask(h, 1 << cpu):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
    if not k.SetPriorityClass(h, 0x80):                      # HIGH_PRIORITY_CLASS
        raise OSError(ctypes.get_last_error(), "SetPriorityClass")
    # Version 1, control the execution speed, state 0: never throttle this process (ProcessPowerThrottling = 4).
    state = PowerThrottlingState(1, 1, 0)
    if not k.SetProcessInformation(h, 4, ctypes.byref(state), ctypes.sizeof(state)):
        raise OSError(ctypes.get_last_error(), "SetProcessInformation")
    return "pinned to logical CPU %d, high priority, power throttling off" % cpu
