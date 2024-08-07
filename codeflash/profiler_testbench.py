class classA:
    @staticmethod
    def funcA(number):
        j = 0
        for i in range(number):
            j += i
        string_comprehension = [str(i) for i in range(number)]
        return " ".join(string_comprehension)


class classB:
    @classmethod
    def funcB(cls, number):
        j = 0
        for i in range(number):
            j += i
        string_concatenation = ""
        for i in range(number):
            string_concatenation += str(i)
        return string_concatenation


def funcC(number):
    if number == 1:
        return "0"
    string_formatting = ""
    for i in range(number):
        string_formatting += f"{i}"
    return string_formatting


def compare_functions(number):
    resultA = classA.funcA(number)
    for _ in range(2000):
        resultB = classB.funcB(number)
        resultC = funcC(1)
        equal = resultA == resultB == resultC


if __name__ == "__main__":
    compare_functions(20)
