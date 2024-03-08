def problem_p02699():
    S, W = list(map(int, input().split()))

    if S <= W:

        print("unsafe")

    elif S > W:

        print("safe")


problem_p02699()
