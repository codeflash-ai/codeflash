def problem_p00500():
    N = int(input())

    score = [list(map(int, input().split())) for _ in range(N)]

    players = [0 for _ in range(N)]

    for play in list(zip(*score)):

        for i, p in enumerate(play):

            if play.count(p) == 1:

                players[i] += p

    print(*players, sep="\n")


problem_p00500()
