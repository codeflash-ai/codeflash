def problem_p02918():
    n, k = list(map(int, input().split()))

    s = eval(input())

    happy = 0

    seq = []

    cur = s[0]

    seq.append(cur)

    for i in range(1, n):

        if s[i] != cur:

            seq.append(s[i])

            cur = s[i]

        else:

            happy = happy + 1

    if len(seq) % 2 == 1:

        rev = len(seq) // 2

        print((happy + min(rev, k) * 2))

    else:

        rev = len(seq) // 2

        if rev > k:

            print((happy + k * 2))

        else:

            print((happy + (rev - 1) * 2 + 1))


problem_p02918()
