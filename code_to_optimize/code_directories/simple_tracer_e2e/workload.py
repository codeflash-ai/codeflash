from concurrent.futures import ThreadPoolExecutor


def funcA(number):
    number = number if number < 100 else 100
    k = 0
    for i in range(number * 10):
        k += i
    j = sum(range(number))
    return " ".join(str(i) for i in range(number))


def test_threadpool() -> None:
    pool = ThreadPoolExecutor(max_workers=2)
    args = [5, 10, 15]
    result = pool.map(funcA, args)

    for r in result:
        print(r)

class AlexNet:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def forward(self, x):
        result = 0
        for val in x:
            result += val * val
        return result % self.num_classes


def test_models():
    model = AlexNet(num_classes=10)
    input_data = [1, 2, 3, 4, 5]
    result = model.forward(input_data)

if __name__ == "__main__":
    test_threadpool()
    test_models()
