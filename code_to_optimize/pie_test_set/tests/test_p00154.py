from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00154_0():
    input_content = "5\n1 10\n5 3\n10 3\n25 2\n50 2\n4\n120\n500\n100\n168\n7\n1 10\n3 10\n5 10\n10 10\n25 10\n50 10\n100 10\n3\n452\n574\n787\n0"
    expected_output = "16\n0\n12\n7\n9789\n13658\n17466"
    run_pie_test_case("../p00154.py", input_content, expected_output)


def test_problem_p00154_1():
    input_content = "5\n1 10\n5 3\n10 3\n25 2\n50 2\n4\n120\n500\n100\n168\n7\n1 10\n3 10\n5 10\n10 10\n25 10\n50 10\n100 10\n3\n452\n574\n787\n0"
    expected_output = "16\n0\n12\n7\n9789\n13658\n17466"
    run_pie_test_case("../p00154.py", input_content, expected_output)
