from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03019_0():
    input_content = "2 100\n85 2 3\n60 1 1"
    expected_output = "115"
    run_pie_test_case("../p03019.py", input_content, expected_output)


def test_problem_p03019_1():
    input_content = "2 100\n85 2 3\n60 1 1"
    expected_output = "115"
    run_pie_test_case("../p03019.py", input_content, expected_output)


def test_problem_p03019_2():
    input_content = "10 1000\n451 4593 6263\n324 310 6991\n378 1431 7068\n71 1757 9218\n204 3676 4328\n840 6221 9080\n684 1545 8511\n709 5467 8674\n862 6504 9835\n283 4965 9980"
    expected_output = "2540"
    run_pie_test_case("../p03019.py", input_content, expected_output)


def test_problem_p03019_3():
    input_content = "2 100\n85 2 3\n60 10 10"
    expected_output = "77"
    run_pie_test_case("../p03019.py", input_content, expected_output)


def test_problem_p03019_4():
    input_content = "1 100000\n31415 2718 2818"
    expected_output = "31415"
    run_pie_test_case("../p03019.py", input_content, expected_output)
