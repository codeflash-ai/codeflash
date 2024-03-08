from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03270_0():
    input_content = "3 3"
    expected_output = "7\n7\n4\n7\n7"
    run_pie_test_case("../p03270.py", input_content, expected_output)


def test_problem_p03270_1():
    input_content = "6 1000"
    expected_output = "149393349\n149393349\n668669001\n668669001\n4000002\n4000002\n4000002\n668669001\n668669001\n149393349\n149393349"
    run_pie_test_case("../p03270.py", input_content, expected_output)


def test_problem_p03270_2():
    input_content = "3 3"
    expected_output = "7\n7\n4\n7\n7"
    run_pie_test_case("../p03270.py", input_content, expected_output)


def test_problem_p03270_3():
    input_content = "4 5"
    expected_output = "36\n36\n20\n20\n20\n36\n36"
    run_pie_test_case("../p03270.py", input_content, expected_output)
