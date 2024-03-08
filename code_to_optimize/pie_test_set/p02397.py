def problem_p02397():
    while 1:

        x, y = map(int, input().split())

        if not x + y:
            break

        print(*[y, x] if x > y else [x, y])


problem_p02397()
