def problem_p03068():
    import re

    N = int(eval(input()))

    S = eval(input())

    K = int(eval(input()))

    w = S[K - 1]

    print((re.sub("[^#]", "*", S.replace(w, "#")).replace("#", w)))


problem_p03068()
