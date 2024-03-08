def problem_p01646():
    def add_edge(node, adj_lst, adj_rev, s1, s2):

        ind = 0

        max_len = min(len(s1), len(s2))

        while ind < max_len and s1[ind] == s2[ind]:

            ind += 1

        if ind == max_len:

            if max_len < len(s1):

                return True

            return False

        c1 = ord(s1[ind]) - ord("a")

        c2 = ord(s2[ind]) - ord("a")

        adj_lst[c1].add(c2)

        adj_rev[c2].add(c1)

        node.add(c1)

        node.add(c2)

        return False

    """
    
    def dfs(x, visited, adj_lst, order):
    
      visited[x] = True
    
      for to in adj_lst[x]:
    
        if not visited[to]:
    
          dfs(to, visited, adj_lst, order)
    
      order.append(x)
    
    
    
    def dfs_rev(x, used, adj_rev):
    
      used.append(x)
    
      for to in adj_rev[x]:
    
        if not to in used:
    
          dfs_rev(x, used, adj_rev)
    
    """

    while True:

        n = int(eval(input()))

        if n == 0:

            break

        lst = [eval(input()) for _ in range(n)]

        node = set()

        adj_lst = [set() for _ in range(26)]

        adj_rev = [set() for _ in range(26)]

        brunk_flag = False

        for i in range(n):

            for j in range(i + 1, n):

                brunk_flag = brunk_flag or add_edge(node, adj_lst, adj_rev, lst[i], lst[j])

        L = []

        visited = [False] * 26

        cycle_flag = False

        def visit(n):

            global cycle_flag

            if cycle_flag:
                return

            if visited[n] == 2:

                cycle_flag = True

            elif visited[n] == 0:

                visited[n] = 2

                for to in adj_lst[n]:

                    visit(to)

                visited[n] = 1

                L.append(n)

        L = []

        for n in node:

            visit(n)

        if cycle_flag or brunk_flag:

            print("no")

        else:

            print("yes")


problem_p01646()
