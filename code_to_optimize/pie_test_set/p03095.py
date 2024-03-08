def problem_p03095():
    N = int(eval(input()))

    S = eval(input())

    mod = 7 + 10**9

    word = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    count = [1] * 26

    for i in range(N):

        count[word.index(S[i])] += 1

    Total = 1

    for c in count:

        Total *= c

        Total %= mod

    print((Total - 1))


problem_p03095()
