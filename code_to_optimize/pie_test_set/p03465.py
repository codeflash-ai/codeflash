def problem_p03465():
    N = int(eval(input()))

    A = [int(a) for a in input().split()]

    s = 1

    for a in A:

        s |= s << a

    for i in range((sum(A) + 1) // 2, sum(A) + 1):

        if s & (1 << i):

            print(i)

            break


problem_p03465()
