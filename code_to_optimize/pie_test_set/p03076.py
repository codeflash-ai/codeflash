def problem_p03076():
    a = eval("int(input())," * 5)
    print((min(~-i % 10 for i in a) - sum(-i // 10 * 10 for i in a) - 9))


problem_p03076()
