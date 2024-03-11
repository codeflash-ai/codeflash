def problem_p03560(input_data):
    k = int(eval(input_data))

    from collections import deque

    dic = {1: 1}

    queue = deque([[1, 1]])

    while queue:

        v, cost = queue.popleft()

        v_ = (v + 1) % k

        if not v_ in dic or cost + 1 < dic[v_]:

            dic[v_] = cost + 1

            queue.appendleft([v_, cost + 1])

        v_ = v * 10 % k

        if not v_ in dic or cost < dic[v_]:

            dic[v_] = cost

            queue.append([v_, cost])

    return dic[0]
