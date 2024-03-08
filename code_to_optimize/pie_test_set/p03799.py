def problem_p03799():
    s, c = list(map(int, input().split()))

    ans = min(s, c // 2)

    c -= ans * 2

    ans += c // 4

    print(ans)


problem_p03799()
