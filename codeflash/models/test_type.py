from enum import Enum


class TestType(Enum):
    EXISTING_UNIT_TEST = 1
    INSPIRED_REGRESSION = 2
    GENERATED_REGRESSION = 3
    REPLAY_TEST = 4
    CONCOLIC_COVERAGE_TEST = 5
    INIT_STATE_TEST = 6

    def to_name(self) -> str:
        if self is TestType.INIT_STATE_TEST:
            return ""
        names = {
            TestType.EXISTING_UNIT_TEST: "⚙️ Existing Unit Tests",
            TestType.INSPIRED_REGRESSION: "🎨 Inspired Regression Tests",
            TestType.GENERATED_REGRESSION: "🌀 Generated Regression Tests",
            TestType.REPLAY_TEST: "⏪ Replay Tests",
            TestType.CONCOLIC_COVERAGE_TEST: "🔎 Concolic Coverage Tests",
        }
        return names[self]
