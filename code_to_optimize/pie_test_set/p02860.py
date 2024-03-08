def problem_p02860():
    n = int(eval(input()))

    s = eval(input())

    print(("Yes" if n % 2 == 0 and s[: n // 2] == s[n // 2 :] else "No"))


problem_p02860()
