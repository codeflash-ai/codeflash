def problem_p02947():
    import sys

    def input():

        return sys.stdin.readline()[:-1]

    N = int(eval(input()))

    sentences = [None] * N

    dp = {}

    for i in range(N):

        k = list(map(str, eval(input())))

        k.sort()

        sentences[i] = k

    def addTree(tree, sentence, org):

        l = len(sentence)

        if l == 0:

            return None

        st = sentence[0]

        if st not in tree:

            tree[st] = {}

        else:

            if l == 1:

                org = "".join(org)

                if org in dp:

                    dp[org] += 1

                else:

                    dp[org] = 2

        tree[sentence[0]] = addTree(tree[st], sentence[1:], org)

        return tree

    def createTree(sentences):

        tree = {}

        for sentence in sentences:

            tree = addTree(tree, sentence, sentence)

        return tree

    tree = createTree(sentences)

    ans = 0

    for i in list(dp.values()):

        ans += i * (i - 1) / 2

    print((int(ans)))


problem_p02947()
