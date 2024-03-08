def problem_p02233():
    def fib2(n):

        a1, a2 = 1, 0

        while n > 0:

            a1, a2 = a1 + a2, a1

            n -= 1

        return a1

    n = int(eval(input()))

    print((fib2(n)))


problem_p02233()
