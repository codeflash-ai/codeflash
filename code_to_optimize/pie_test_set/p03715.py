def problem_p03715(input_data):
    # coding: utf-8

    # Your code here!

    h, w = [int(i) for i in input_data.split()]

    def once(h, w):

        res = 10**9

        if h > 1:

            a, b = h // 2 * w, (h - h // 2) * w

            res = abs(a - b)

        if w > 1:

            c, d = w // 2 * h, (w - w // 2) * h

            if abs(c - d) < res:

                a, b = c, d

        return a, b

    def hsaki(h, w):

        res = 10**9

        for i in range(1, h):

            p1 = i * w

            p2, p3 = once(h - i, w)

            a = max(p1, p2, p3) - min(p1, p2, p3)

            res = min(res, a)

        #        return (p1,p2,p3)

        return res

    # return (hsaki(h,w), hsaki(w,h))

    ans = min(hsaki(h, w), hsaki(w, h))

    return ans
