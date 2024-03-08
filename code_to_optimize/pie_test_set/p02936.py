def problem_p02936():
    from pprint import pprint

    from collections import deque, defaultdict

    import sys

    # n, q = map(int, input().strip().split(" "))

    n, q = list(map(int, sys.stdin.readline().strip().split(" ")))

    edges = [[] for _ in range(n)]

    for _ in range(n - 1):

        # a_i, b_i = map(int, input().strip().split(" "))

        a_i, b_i = list(map(int, sys.stdin.readline().strip().split(" ")))

        edges[a_i - 1].append(b_i - 1)

        edges[b_i - 1].append(a_i - 1)

    counter = [0] * n

    p = [0] * n

    for _ in range(q):

        # p_i, x_i = map(int, input().strip().split(" "))

        p_i, x_i = list(map(int, sys.stdin.readline().strip().split(" ")))

        p[p_i - 1] += x_i

        # counter[p_i-1] += x_i

    parents = deque()

    parents.append(0)

    visited = set()

    while parents:

        parent = parents.popleft()

        if parent in visited:

            continue

        counter[parent] += p[parent]

        visited.add(parent)

        for child in edges[parent]:

            if child in visited:

                continue

            counter[child] += counter[parent]

            # print("child")

            # print(child, counter[child])

            parents.append(child)

    print((" ".join(list(map(str, counter)))))


problem_p02936()
