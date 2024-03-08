def problem_p03534():
    S = eval(input())

    C = {"a": 0, "b": 0, "c": 0}

    for s in S:

        C[s] += 1

    if max(C.values()) - min(C.values()) <= 1:

        print("YES")

    else:

        print("NO")


problem_p03534()
