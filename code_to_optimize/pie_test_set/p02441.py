def problem_p02441():
    # AOJ ITP2_3_C: Count

    # Python3 2018.6.24 bal4u

    n = int(eval(input()))

    a = list(map(int, input().split()))

    q = int(eval(input()))

    for i in range(q):

        b, e, k = list(map(int, input().split()))

        s = a[b:e]

        print((s.count(k)))


problem_p02441()
