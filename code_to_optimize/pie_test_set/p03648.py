def problem_p03648():
    b = eval(input())
    a = b / 50 + 49
    b %= 50
    c = 50 - b
    print("50\n" + (repr(a - b) + " ") * c + (repr(a + c) + " ") * b)


problem_p03648()
