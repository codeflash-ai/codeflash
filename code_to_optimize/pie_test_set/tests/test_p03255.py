from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03255_0():
    input_content = "2 100\n1 10"
    expected_output = "355"
    run_pie_test_case("../p03255.py", input_content, expected_output)


def test_problem_p03255_1():
    input_content = "5 1\n1 999999997 999999998 999999999 1000000000"
    expected_output = "19999999983"
    run_pie_test_case("../p03255.py", input_content, expected_output)


def test_problem_p03255_2():
    input_content = "16 10\n1 7 12 27 52 75 731 13856 395504 534840 1276551 2356789 9384806 19108104 82684732 535447408"
    expected_output = "3256017715"
    run_pie_test_case("../p03255.py", input_content, expected_output)


def test_problem_p03255_3():
    input_content = "2 100\n1 10"
    expected_output = "355"
    run_pie_test_case("../p03255.py", input_content, expected_output)


def test_problem_p03255_4():
    input_content = "10 8851025\n38 87 668 3175 22601 65499 90236 790604 4290609 4894746"
    expected_output = "150710136"
    run_pie_test_case("../p03255.py", input_content, expected_output)
