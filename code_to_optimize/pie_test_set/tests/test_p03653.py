from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03653_0():
    input_content = "1 2 1\n2 4 4\n3 2 1\n7 6 7\n5 2 3"
    expected_output = "18"
    run_pie_test_case("../p03653.py", input_content, expected_output)


def test_problem_p03653_1():
    input_content = "1 2 1\n2 4 4\n3 2 1\n7 6 7\n5 2 3"
    expected_output = "18"
    run_pie_test_case("../p03653.py", input_content, expected_output)


def test_problem_p03653_2():
    input_content = "6 2 4\n33189 87907 277349742\n71616 46764 575306520\n8801 53151 327161251\n58589 4337 796697686\n66854 17565 289910583\n50598 35195 478112689\n13919 88414 103962455\n7953 69657 699253752\n44255 98144 468443709\n2332 42580 752437097\n39752 19060 845062869\n60126 74101 382963164"
    expected_output = "3093929975"
    run_pie_test_case("../p03653.py", input_content, expected_output)


def test_problem_p03653_3():
    input_content = "3 3 2\n16 17 1\n2 7 5\n2 16 12\n17 7 7\n13 2 10\n12 18 3\n16 15 19\n5 6 2"
    expected_output = "110"
    run_pie_test_case("../p03653.py", input_content, expected_output)
