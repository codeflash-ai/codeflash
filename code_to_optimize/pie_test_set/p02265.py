def problem_p02265():
    from collections import deque

    queue = deque()

    for _ in range(int(eval(input()))):

        commands = input().split(" ")

        if commands[0] == "insert":

            queue.appendleft(commands[1])

        elif commands[0] == "delete":

            try:

                queue.remove(commands[1])

            except ValueError:

                pass

        elif commands[0] == "deleteFirst":

            queue.popleft()

        elif commands[0] == "deleteLast":

            queue.pop()

    print((" ".join(queue)))


problem_p02265()
