def problem_p03808():
    n = int(eval(input()))

    lis = list(map(int, input().split()))

    num = (n * (n + 1)) // 2

    ll = sum(lis)

    if ll % num != 0:

        print("NO")

        exit()

    ans = 0

    for i in range(n):

        if abs(lis[i] - lis[i - 1] - ll // num) % n != 0:

            print("NO")

            exit()

        ans += abs(lis[i] - lis[i - 1] - ll // num) // n

    if ans == ll // num:
        print("YES")

    else:
        print("NO")


problem_p03808()
