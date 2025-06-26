from concurrent.futures import ThreadPoolExecutor


def funcA(number):
    # Clamp the number to a maximum of 1000
    number = min(1000, number)

    # Use arithmetic sum for much faster calculation
    k = (number * 100) * (number * 100 - 1) // 2

    # Use arithmetic sum for much faster calculation
    j = number * (number - 1) // 2

    # Use list comprehension with join; for large numbers, this uses less time than a generator
    return " ".join([str(i) for i in range(number)])


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
        # Since _extract_features always returns [], directly pass []
        return self._classify([])

    def _extract_features(self, x):
        # The original loop did nothing; just return an empty list immediately
        return []

    def _classify(self, features):
        # Since features is always [], sum(features) == 0, len(features) == 0
        total_mod = 0 % self.num_classes
        return []  # Directly return empty list since len(features) == 0


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


if __name__ == "__main__":
    test_threadpool()
    test_models()
