def problem_p03792():
    n = int(eval(input()))
    a = eval("list(input())," * n)
    f = sum(n == t.count("#") for t in zip(*a))
    print(
        (
            -all(t.count("#") < 1 for t in a)
            or min(
                n * 2 - f - a[i].count("#") + all(a[j][i] > "#" for j in range(n)) for i in range(n)
            )
        )
    )


problem_p03792()
