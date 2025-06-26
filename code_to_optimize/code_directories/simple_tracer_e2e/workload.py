from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


def funcA(number):
    number = min(1000, number)
    j = number * (number - 1) // 2
    return _joined_numbers(number)


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
        # preallocate the empty list with correct size for potential future use, but currently just pass
        return []

    def _classify(self, features):
        # Compute total sum once, then compute modulo only once.
        # Use list multiplication for fast vector creation.
        if not features:
            return []
        mod_val = sum(features) % self.num_classes
        return [mod_val] * len(features)


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


@lru_cache(maxsize=32)
def _joined_numbers(n):
    return " ".join(map(str, range(n)))


if __name__ == "__main__":
    test_threadpool()
    test_models()
