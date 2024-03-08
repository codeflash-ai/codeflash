def problem_p01131():
    d = {
        "1": [".", ",", "!", "?", " "],
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"],
    }

    n = int(eval(input()))

    m = [eval(input()) for _ in range(n)]

    for line in m:

        ans = []

        for cs in line.split("0"):

            if cs == "":

                continue

            index = len(cs) % len(d[cs[0]]) - 1

            ans.append(d[cs[0]][index])

        print(("".join(ans)))


problem_p01131()
