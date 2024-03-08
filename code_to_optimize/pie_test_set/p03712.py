def problem_p03712():
    h, w = (int(x) for x in input().split())

    ab = []

    ab.append(["#"] * (w + 2))

    for _ in range(h):

        s = eval(input())

        s = list(s)

        s = ["#"] + s + ["#"]

        ab.append(s)

    ab.append(["#"] * (w + 2))

    for i in range(h + 2):

        k = ab[i]

        print(("".join(map(str, k))))


problem_p03712()
