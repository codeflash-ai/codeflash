def problem_p02628():
    x, y = list(map(int, input().split()))

    n = sorted(list(map(int, input().split())))

    print((sum(n[:y])))


problem_p02628()
