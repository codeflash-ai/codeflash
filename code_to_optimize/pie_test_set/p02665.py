def problem_p02665():
    n = int(eval(input()))

    leaves = list(map(int, input().split()))

    ans = 0

    nodes_min = [0] * (n + 1)

    nodes_max = [0] * (n + 1)

    nodes_min[n] = leaves[n]

    nodes_max[n] = leaves[n]

    for depth in range(n, 0, -1):

        root_min = nodes_min[depth] // 2 + nodes_min[depth] % 2

        nodes_min[depth - 1] = leaves[depth - 1] + root_min

        root_max = nodes_max[depth]

        nodes_max[depth - 1] = leaves[depth - 1] + root_max

    nodes = [0] * (n + 1)

    nodes[0] = 1

    if nodes[0] < nodes_min[0]:

        print((-1))

    else:

        for depth in range(n):

            roots = nodes[depth] - leaves[depth]

            nodes[depth + 1] = min(roots * 2, nodes_max[depth + 1])

            if nodes[depth + 1] < nodes_min[depth + 1]:

                print((-1))

                break

        else:

            print((sum(nodes)))


problem_p02665()
