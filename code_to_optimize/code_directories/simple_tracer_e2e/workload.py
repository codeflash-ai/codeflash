from concurrent.futures import ThreadPoolExecutor


def funcA(number):
    number = min(1000, number)
    k = 0
    for i in range(number * 100):
        k += i
    # Simplify the for loop by using sum with a range object
    j = sum(range(number))

    # Use a generator expression directly in join for more efficiency
    return " ".join(str(i) for i in range(number))


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
        # Directly return [] since _extract_features returns [] always
        return []

    def _extract_features(self, x):
        # Return empty list immediately
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


if __name__ == "__main__":
    test_threadpool()
    test_models()
