# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.

import ctypes
from pathlib import Path
import threading

_active = set()
_lock = threading.Lock()


class CudaWorkspace:
    @staticmethod
    def claim(path):
        key = str(Path(path).resolve()).casefold()
        with _lock:
            if key in _active:
                raise RuntimeError('This bundle already has an active CUDA session')
            _active.add(key)
        return key

    @staticmethod
    def release_claim(key):
        with _lock:
            _active.discard(key)

    def __init__(self, path, device_id, maximum_bytes=1024**3, claimed_key=None):
        self.key = claimed_key or self.claim(path)
        self.closed = False
        self.device_id = device_id
        self.maximum_bytes = maximum_bytes
        self.owners = {}
        self.free = set()
        self.errors = []
        self.calls = 0
        self.bytes = 0
        try:
            self.dll = ctypes.CDLL(str(path))
            self.driver = ctypes.WinDLL('nvcuda.dll')
            self.driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
            self.driver.cuMemAlloc_v2.restype = ctypes.c_int
            self.driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
            self.driver.cuMemFree_v2.restype = ctypes.c_int
            alloc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_uint64, ctypes.c_int, ctypes.c_int)
            free_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
            def context_slot(name):
                try:
                    return ctypes.c_void_p.in_dll(self.dll, name)
                except ValueError:
                    return None
            self.alloc_slot = context_slot('__TVMBackendAllocWorkspace')
            self.free_slot = context_slot('__TVMBackendFreeWorkspace')
            if self.alloc_slot is None and self.free_slot is None:
                return
            if self.alloc_slot is None or self.free_slot is None:
                raise ValueError('Incomplete TVM workspace context ABI')
            self.saved = (self.alloc_slot.value, self.free_slot.value)
            self.original_alloc = alloc_type(self.saved[0])
            self.original_free = free_type(self.saved[1])
            self.allocate = alloc_type(self._allocate)
            self.release = free_type(self._release)
            self.alloc_slot.value = ctypes.cast(self.allocate, ctypes.c_void_p).value
            self.free_slot.value = ctypes.cast(self.release, ctypes.c_void_p).value
        except BaseException:
            self.release_claim(self.key)
            raise

    def _allocate(self, kind, device, size, code, bits):
        try:
            if kind != 2:
                return self.original_alloc(kind, device, size, code, bits)
            if device != self.device_id or self.closed:
                raise RuntimeError('Workspace device or lifetime mismatch')
            candidates = [p for p in self.free if self.owners[p] >= size]
            if candidates:
                pointer = min(candidates, key=self.owners.__getitem__)
                self.free.remove(pointer)
            else:
                size = max(256, int(size))
                if self.bytes + size > self.maximum_bytes:
                    raise MemoryError('CUDA workspace exceeds its admitted reserve')
                memory = ctypes.c_uint64()
                result = self.driver.cuMemAlloc_v2(ctypes.byref(memory), size)
                if result:
                    raise RuntimeError('CUDA workspace allocation failed: ' + str(result))
                pointer = memory.value
                self.owners[pointer] = size
                self.bytes += size
            self.calls += 1
            return pointer
        except BaseException as exc:
            self.errors.append(str(exc))
            return None

    def _release(self, kind, device, pointer):
        try:
            if kind != 2:
                return self.original_free(kind, device, pointer)
            if device != self.device_id or pointer not in self.owners or pointer in self.free:
                raise RuntimeError('Invalid CUDA workspace release')
            self.free.add(pointer)
            return 0
        except BaseException as exc:
            self.errors.append(str(exc))
            return -1

    def check(self):
        if self.errors:
            raise RuntimeError('; '.join(self.errors))

    def close(self):
        if self.closed:
            return
        if self.alloc_slot is not None:
            self.alloc_slot.value, self.free_slot.value = self.saved

        if len(self.free) != len(self.owners):
            self.errors.append('CUDA workspace was still live at session close')
        for pointer in list(self.owners):
            code = self.driver.cuMemFree_v2(pointer)
            if code:
                self.errors.append('CUDA workspace release failed: ' + str(code))
            else:
                del self.owners[pointer]
        self.free.clear()
        self.closed = True
        self.release_claim(self.key)
        self.check()
