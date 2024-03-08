def problem_p00179():
    from collections import deque

    def new_color(s, i, rgb):

        for color in rgb:

            if color != s[i] and color != s[i + 1]:

                break

        return s[:i] + color * 2 + s[i + 2 :]

    def solve(s):

        length = len(s)

        monos = ["r" * length, "g" * length, "b" * length]

        if s in monos:

            print((0))

            return

        dic = {s: 0}

        rgb = "rgb"

        que = deque()

        app = que.append

        pop = que.popleft

        que.append((s, 0))

        while que:

            colors, score = pop()

            score += 1

            temp = colors[0]

            for i in range(1, length):

                ci = colors[i]

                if ci != temp:

                    new = new_color(colors, i - 1, rgb)

                    if new in monos:

                        print(score)

                        return

                    if new not in dic:

                        dic[new] = score

                        app((new, score))

                temp = ci

        print("NA")

    def main():

        while True:

            s = eval(input())

            if s == "0":

                break

            solve(s)

    main()


problem_p00179()
