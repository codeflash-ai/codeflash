def problem_p03478():
    N, A, B = list(map(int, input().split()))

    ans = 0

    for i in range(1, N + 1):

        if A <= sum(list(map(int, str(i)))) <= B:

            ans += i

    print(ans)


problem_p03478()
