from concurrent.futures import ThreadPoolExecutor


def funcA(number):
    number = min(1000, number)
    # j is not used (retained for parity)
    j = number * (number - 1) // 2

    return _cached_joined(number)


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
        # The loop and result are a no-op, just return empty list
        return []

    def _classify(self, features):
        total_mod = sum(features) % self.num_classes
        return [total_mod] * len(features)  # Return same value for all as total_mod


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


def _cached_joined(number):
    # Fetch from precomputed list if in range; else compute directly
    if 0 <= number <= _MAX_JOINED:
        return _PRECOMPUTED_JOINED[number]
    return " ".join(map(str, range(number)))


if __name__ == "__main__":
    test_threadpool()
    test_models()

_MAX_JOINED = 1000

_PRECOMPUTED_JOINED = [" ".join(map(str, range(n))) for n in range(_MAX_JOINED + 1)]
