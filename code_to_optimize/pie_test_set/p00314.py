def problem_p00314():
    from itertools import takewhile as T

    N = int(eval(input()))

    p = list(map(int, input().split()))

    score = [0] * 101

    for x in p:

        score[x] += 1

    print((len(list(T(lambda i: sum(score[i:]) >= i, list(range(1, 101)))))))


problem_p00314()
