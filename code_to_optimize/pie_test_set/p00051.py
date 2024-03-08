def problem_p00051():
    n = eval(input())

    while n:

        n -= 1

        a = "".join(sorted(str(eval(input()))))

        print(int(a[::-1]) - int(a))


problem_p00051()
