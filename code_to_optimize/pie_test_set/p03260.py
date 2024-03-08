def problem_p03260():
    a, b = list(map(int, input().split()))

    print(("Yes" if (a * b) % 2 else "No"))


problem_p03260()
