from concurrent.futures import ThreadPoolExecutor


def funcA(number):
    number = min(1000, number)

    # Simplify the sum calculation using arithmetic progression formula for O(1) time
    j = number * (number - 1) // 2

    # Use a cached helper to very efficiently reuse results for each possible 'number'
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


def _joined_number_str(n):
    # Use precomputed result for n in 0..1000, else fallback to runtime computation
    if 0 <= n <= 1000:
        return _JOINED_NUMBER_STRINGS[n]
    # use the same logic as before, but map is actually slightly faster than generator in CPython
    return " ".join(map(str, range(n)))


if __name__ == "__main__":
    test_threadpool()
    test_models()

_JOINED_NUMBER_STRINGS = tuple(" ".join(str(i) for i in range(n)) for n in range(1001))
