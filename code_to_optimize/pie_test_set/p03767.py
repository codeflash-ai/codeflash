def problem_p03767():
    N = eval(input())

    arr = list(map(int, input().split()))

    arr.sort()

    sum = 0

    for i in range(3 * N - 2, N - 1, -2):

        sum += arr[i]

    print(sum)


problem_p03767()
