def problem_p02282():
    def post_from_pre_in(preorder_elements, inorder_elements):

        if preorder_elements:

            global cnt

            root = preorder_elements[0]

            root_index = inorder_elements.index(root)

            preorder_left = preorder_elements[1 : root_index + 1]

            inorder_left = inorder_elements[:root_index]

            preorder_right = preorder_elements[root_index + 1 :]

            inorder_right = inorder_elements[root_index + 1 :]

            post_from_pre_in(preorder_left, inorder_left)

            post_from_pre_in(preorder_right, inorder_right)

            if cnt:

                print(root, end=" ")

                cnt -= 1

            else:

                print(root)

    cnt = int(input()) - 1

    a = list(map(int, input().split()))

    b = list(map(int, input().split()))

    post_from_pre_in(a, b)


problem_p02282()
