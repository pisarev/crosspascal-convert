# Copyright (C) 2026 Yuriy Pisarev. MIT license; see LICENSE.
from contextlib import contextmanager


@contextmanager
def cancellable_cpu_call(vm, should_stop):
    if should_stop is None:
        yield
        return
    cancelled = False

    def check(func, symbol, before, result, *args):
        nonlocal cancelled
        if before and should_stop():
            cancelled = True
            raise InterruptedError('Operation cancelled')
        return 0

    vm.set_instrument(check)
    try:
        yield
    except Exception:
        if cancelled:
            raise InterruptedError('Operation cancelled') from None
        raise
    finally:
        vm.set_instrument(None)
