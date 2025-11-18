import time
from typing import Any


class SplitTimer:
    def __init__(self, prefix: str = '') -> None:
        self.mark: int = time.process_time_ns()
        self.start = self.mark
        self.prefix = f'{prefix} -' if prefix else ''

    def reset(self) -> None:
        self.mark = time.process_time_ns()

    def print(self, *message: Any) -> None:
        process_time = time.process_time_ns()

        total = process_time - self.start
        split = process_time - self.mark

        print(
            self.prefix,
            self.format_time(total),
            '/',
            self.format_time(split),
            '-',
            *message)

        self.mark = time.process_time_ns()

    @staticmethod
    def format_time(time: int) -> str:
        units = ['μs', 'ns', 'ms', 's']
        time_f = float(time)
        while time_f > 1000 and len(units) > 1:
            time_f = time_f / 1000
            units.pop(0)
        unit = units.pop(0)
        return f'{time_f:7.3f}{unit: <2}'
