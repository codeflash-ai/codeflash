def problem_p02589():
    from collections import Counter

    n = int(eval(input()))

    sss = [eval(input()) for _ in range(n)]

    d = {s[1:]: 0 for s in sss}

    flags = [1 << (i * 18) for i in range(26)]

    trie_chr = [-1]

    trie_children = [{}]

    trie_counter = [0]

    trie_fin = [0]

    s_fin_node = []

    l = 1

    for s in sss:

        node = 0

        trie_counter[node] += 1

        for i in range(len(s) - 1, 0, -1):

            c = ord(s[i]) - 97

            if c in trie_children[node]:

                node = trie_children[node][c]

                trie_counter[node] += 1

            else:

                trie_chr.append(c)

                trie_children.append({})

                trie_fin.append(0)

                trie_children[node][c] = l

                trie_counter.append(1)

                node = l

                l += 1

        trie_fin[node] += 1

        s_fin_node.append(node)

    trie_flags = [0] * l

    for s in sss:

        cnt = Counter(s)

        flag = 0

        for c in cnt:

            flag |= flags[ord(c) - 97]

        node = 0

        if len(s) > 1 and trie_fin[node] > 0:

            trie_flags[node] += flag

        for i in range(len(s) - 1, 1, -1):

            c = s[i]

            k = ord(c) - 97

            cnt[c] -= 1

            if cnt[c] == 0:

                flag ^= flags[k]

            node = trie_children[node][k]

            # print(s, i, node, trie_counter[node])

            if trie_counter[node] == 1:

                break

            if trie_fin[node] > 0:

                trie_flags[node] += flag

                # print('add', s, i)

    # print([chr(c) if c >= 0 else -1 for c in trie_chr])

    # print(trie_counter)

    # print(trie_fin)

    # print(trie_flags)

    # print(s_fin_node)

    ans = 0

    mask = (1 << 18) - 1

    for i, s in enumerate(sss):

        cnt = (trie_flags[s_fin_node[i]] >> ((ord(s[0]) - 97) * 18)) & mask

        ans += cnt

    print(ans)


problem_p02589()
