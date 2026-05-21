from __future__ import annotations


class LinearSchedule:
    def __init__(self, start: float, end: float, decay_steps: int) -> None:
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        frac = min(max(int(step), 0) / self.decay_steps, 1.0)
        return self.start + frac * (self.end - self.start)

    def __call__(self, step: int) -> float:
        return self.value(step)


class ConstantSchedule:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self, step: int) -> float:
        return self._value

    def __call__(self, step: int) -> float:
        return self.value(step)


def make_schedule(config: dict | None, default: float = 0.0):
    if config is None:
        return ConstantSchedule(default)
    schedule_type = config.get("type", "constant")
    if schedule_type == "linear":
        return LinearSchedule(
            start=config.get("start", default),
            end=config.get("end", default),
            decay_steps=config.get("decay_steps", 1),
        )
    if schedule_type == "constant":
        return ConstantSchedule(config.get("value", config.get("start", default)))
    raise ValueError(f"Unsupported schedule type: {schedule_type}")
