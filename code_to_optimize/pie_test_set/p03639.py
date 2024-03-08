def problem_p03639():
    from collections import defaultdict

    N = int(input())

    counter = defaultdict(int)

    for a in map(int, input().split()):

        if a % 4 == 0:

            counter[2] += 1

        elif a % 2 == 0:

            counter[1] += 1

        else:

            counter[0] += 1

    if counter[1] == 0:

        print("Yes") if counter[0] <= counter[2] + 1 else print("No")

    else:

        print("Yes") if counter[0] <= counter[2] else print("No")


problem_p03639()
