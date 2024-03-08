def problem_p03086():
    import re

    S = eval(input())

    max = 0

    for i in range(len(S)):

        for j in range(i + 1, len(S) + 1):

            pattern = re.compile(r"[A|T|C|G]{%d}" % int(j - i))

            subStr = S[i:j]

            if pattern.match(subStr) and j - i > max:

                max = j - i

    print(max)


problem_p03086()
