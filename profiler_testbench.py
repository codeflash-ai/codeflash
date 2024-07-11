def funcA(number):
    string_comprehension = [str(i) for i in range(number)]
    return " ".join(string_comprehension)


def funcB(number):
    string_concatenation = ""
    for i in range(number):
        string_concatenation += str(i)
    return string_concatenation


def funcC(number):
    string_formatting = ""
    for i in range(number):
        string_formatting += f"{i}"
    return string_formatting


def compare_functions(number):
    for _ in range(2000):
        resultA = funcA(number)
        resultB = funcB(number)
        resultC = funcC(number)
        equal = resultA == resultB == resultC


if __name__ == "__main__":
    compare_functions(20)
