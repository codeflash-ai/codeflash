from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


def funcA(number):
    if number <= 0:
        return ""
    number = min(number, 1000)
    return _joined_number_str(number)


def test_threadpool() -> None:
    pool = ThreadPoolExecutor(max_workers=3)
    args = list(range(10, 31, 10))
    result = pool.map(funcA, args)

    for r in result:
        print(r)


class AlexNet:
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
        self.features_size = 256 * 6 * 6

    def forward(self, x):
        features = self._extract_features(x)

        output = self._classify(features)
        return output

    def _extract_features(self, x):
        # The original implementation does nothing and returns an empty list.
        # Optimized to directly return an empty list.
        return []

    def _classify(self, features):
        total = sum(features)
        return [total % self.num_classes for _ in features]


class SimpleModel:
    @staticmethod
    def predict(data):
        return [x * 2 for x in data]

    @classmethod
    def create_default(cls):
        return cls()


def test_models():
    model = AlexNet(num_classes=10)
    input_data = [1, 2, 3, 4, 5]
    result = model.forward(input_data)

    model2 = SimpleModel.create_default()
    prediction = model2.predict(input_data)


@lru_cache(maxsize=1001)
def _joined_number_str(n):
    # Handle base cases fast
    if n == 0:
        return ""
    if n == 1:
        return "0"
    # Use a list comprehension for string conversion, which is slightly faster than map in Py3
    s_list = [str(i) for i in range(n)]
    return " ".join(s_list)


if __name__ == "__main__":
    test_threadpool()
    test_models()
