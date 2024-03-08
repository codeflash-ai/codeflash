def problem_p00433():
    A = list(map(int, input().split()))

    B = list(map(int, input().split()))

    max_sum = max([sum(A), sum(B)])

    print(max_sum)


problem_p00433()
