def problem_p02255():
    n = int(input())

    arr = list(map(int, input().split(" ")))

    p = 1

    print(" ".join(map(str, arr)))

    while True:

        if p >= len(arr):

            break

        for j in range(0, p):

            if arr[p] < arr[j]:

                arr[p], arr[j] = arr[j], arr[p]

        p += 1

        print(" ".join(map(str, arr)))


problem_p02255()
