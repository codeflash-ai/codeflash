def problem_p03972():
    W, H = list(map(int, input().split()))

    P = [int(eval(input())) for i in range(W)]

    Q = [int(eval(input())) for j in range(H)]

    P.append(float("inf"))

    Q.append(float("inf"))

    P.sort(reverse=True)

    Q.sort(reverse=True)

    now_W = W + 1

    now_H = H + 1

    cost = 0

    while now_W > 1 or now_H > 1:

        if P[-1] < Q[-1]:

            cost += P.pop() * now_H

            now_W -= 1

        elif Q[-1] < P[-1]:

            cost += Q.pop() * now_W

            now_H -= 1

        elif P[-1] == Q[-1]:

            if now_W <= now_H:

                cost += Q.pop() * now_W

                now_H -= 1

            elif now_W > now_H:

                cost += P.pop() * now_H

                now_W -= 1

    print(cost)


problem_p03972()
