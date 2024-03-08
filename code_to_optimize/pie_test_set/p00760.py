def problem_p00760():
    import sys

    if sys.version_info[0] >= 3:
        raw_input = input

    n = int(input())

    for i in range(0, n):

        a = list(map(int, input().split()))

        a[0] -= 1
        a[1] -= 1

        print(
            (
                196471
                - a[0] * 195
                - a[0] / 3 * 5
                - a[1] * 20
                + (a[1] / 2 if a[0] % 3 != 2 else 0)
                - a[2]
            )
        )


problem_p00760()
