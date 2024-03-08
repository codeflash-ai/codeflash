def problem_p03024():
    S = eval(input())

    print(("YES" if 7 - len(S) >= -sum([c == "o" for c in S]) else "NO"))


problem_p03024()
