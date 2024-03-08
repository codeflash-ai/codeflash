def problem_p02898():
    N, K = list(map(int, input().split()))

    h = list(map(int, input().split()))

    h = sorted(h, reverse=True)

    for i, h_i in enumerate(h):

        if h_i >= K:

            continue

        else:

            print(i)

            exit()

    print(N)


problem_p02898()
