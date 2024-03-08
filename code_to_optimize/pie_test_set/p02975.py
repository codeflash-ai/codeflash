def problem_p02975():
    from collections import defaultdict

    n = int(eval(input()))

    a = tuple(map(int, input().split()))

    can = True

    if n % 3 == 0:

        d = defaultdict(int)

        for aa in a:

            d[aa] += 1

        m = n // 3

        t = []

        for number, cnt in list(d.items()):

            if cnt % m != 0:

                can = False

                break

            else:

                for _ in range(cnt // m):

                    t.append(number)

        else:

            if any(t[(0 + i) % 3] ^ t[(1 + i) % 3] != t[(2 + i) % 3] for i in range(3)):

                can = False

    else:

        if any(aa != 0 for aa in a):

            can = False

    print(("Yes" if can else "No"))


problem_p02975()
