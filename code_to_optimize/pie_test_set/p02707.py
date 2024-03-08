def problem_p02707():
    from collections import Counter

    N = int(eval(input()))

    List = list(map(int, input().split()))

    C = Counter(List)

    for i in range(1, N):

        if i in list(C.keys()):

            print((C[i]))

        else:

            print((0))

    print((0))


problem_p02707()
