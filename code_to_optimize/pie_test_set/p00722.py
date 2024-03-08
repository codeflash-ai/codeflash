def problem_p00722():
    prime = [False] * (10**6)

    prime[0] = True

    prime[1] = True

    for i in range(2, 10**6):

        if prime[i] == False:

            for j in range(i * 2, 10**6, i):

                prime[j] = True

    while True:

        a, d, n = list(map(int, input().split()))

        if a == 0:

            break

        cnt = 0

        for i in range(a, 10**6, d):

            if prime[i] == False:

                cnt += 1

            if cnt == n:

                print(i)

                break


problem_p00722()
