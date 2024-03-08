def problem_p02953():
    n = int(eval(input()))

    # n,p=map(int,input().split())

    hl = list(map(int, input().split()))

    # l=[list(map(int,input().split())) for i in range(n)]

    # 素因数分解した結果を2次元配列にして返す

    mx = hl[0]

    ans = "Yes"

    for i in range(n):

        mx = max(mx, hl[i])

        if hl[i] >= mx - 1:

            pass

        else:

            ans = "No"

            break

    print(ans)


problem_p02953()
