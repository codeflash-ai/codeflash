# Format Code for AI Service Request

## Context

You are working on the codeflash optimization engine. The AI service accepts optimization requests with source code and dependency context. A function `calculate_total` in `analytics/metrics.py` needs to be optimized. It calls a helper `normalize_values` in the same file (both modifiable), and imports `BaseMetric` from `analytics/base.py` (not modifiable, just for reference).

```python
# analytics/metrics.py
from analytics.base import BaseMetric

def normalize_values(data: list[float]) -> list[float]:
    max_val = max(data)
    return [x / max_val for x in data]

def calculate_total(metrics: list[BaseMetric]) -> float:
    values = [m.value for m in metrics]
    normalized = normalize_values(values)
    return sum(normalized)
```

```python
# analytics/base.py
class BaseMetric:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
```

## Task

Write a Python function `prepare_optimization_payload` that constructs the code payload for an AI service optimization request for `calculate_total`. It should properly format the source code and dependency code, and include a function to parse the AI service response back into structured code objects.

## Expected Outputs

- A Python file `payload_builder.py` with the payload construction and response parsing logic
