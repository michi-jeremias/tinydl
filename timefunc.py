import shutil
import time
import functools


def timefunc(func):
    """Prints the execution time of a function.

    Keyword arguments:
    func -- the function to be timed
    """

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f'Function: {func.__name__}, time: {time_elapsed:.2f}s')
        return result
    return time_closure


def del_logs(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError as e:
        print('Already deleted.')
