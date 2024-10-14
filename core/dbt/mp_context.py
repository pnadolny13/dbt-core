import threading
import queue


class ThreadingSpawnContextMock:
    def __init__(self):
        self.Lock = threading.Lock
        self.RLock = threading.RLock
        self.Semaphore = threading.Semaphore
        self.BoundedSemaphore = threading.BoundedSemaphore
        self.Event = threading.Event
        self.Condition = threading.Condition
        self.Queue = queue.Queue
        self.JoinableQueue = queue.Queue  # No direct equivalent, using Queue
        self.SimpleQueue = queue.SimpleQueue

    def Process(self, *args, **kwargs):
        return threading.Thread(*args, **kwargs)

    def Pool(self, *args, **kwargs):
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor(*args, **kwargs)

    def Pipe(self):
        class PipeMock:
            def __init__(self):
                self.queue = queue.Queue()

            def send(self, obj):
                self.queue.put(obj)

            def recv(self):
                return self.queue.get()

            def close(self):
                pass  # No direct equivalent for closing a queue

        return PipeMock(), PipeMock()

    def Value(self, typecode, value):
        class ValueMock:
            def __init__(self, value):
                self._value = value
                self._lock = threading.Lock()

            def get(self):
                with self._lock:
                    return self._value

            def set(self, value):
                with self._lock:
                    self._value = value

        return ValueMock(value)

    def Array(self, typecode, sequence):
        class ArrayMock:
            def __init__(self, sequence):
                self._array = list(sequence)
                self._lock = threading.Lock()

            def __getitem__(self, index):
                with self._lock:
                    return self._array[index]

            def __setitem__(self, index, value):
                with self._lock:
                    self._array[index] = value

            def __len__(self):
                return len(self._array)

        return ArrayMock(sequence)


_THREADING_SPAWN_CONTEXT = ThreadingSpawnContextMock()


def get_mp_context():
    return _THREADING_SPAWN_CONTEXT