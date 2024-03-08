def problem_p03130():
    from collections import Counter

    lis = [0 for i in range(4)]

    for i in range(3):

        a, b = list(map(int, input().split()))

        lis[a - 1] += 1

        lis[b - 1] += 1

    if lis.count(2) == 2:

        print("YES")

    else:
        print("NO")


problem_p03130()
