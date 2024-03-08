from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02257_0():
    input_content = "5\n2\n3\n4\n5\n6"
    expected_output = "3"
    run_pie_test_case("../p02257.py", input_content, expected_output)


def test_problem_p02257_1():
    input_content = "5\n2\n3\n4\n5\n6"
    expected_output = "3"
    run_pie_test_case("../p02257.py", input_content, expected_output)


def test_problem_p02257_2():
    input_content = "11\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17"
    expected_output = "4"
    run_pie_test_case("../p02257.py", input_content, expected_output)
