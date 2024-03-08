from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02648_0():
    input_content = "3\n1 2\n2 3\n3 4\n3\n1 1\n2 5\n3 5"
    expected_output = "0\n3\n3"
    run_pie_test_case("../p02648.py", input_content, expected_output)


def test_problem_p02648_1():
    input_content = "15\n123 119\n129 120\n132 112\n126 109\n118 103\n115 109\n102 100\n130 120\n105 105\n132 115\n104 102\n107 107\n127 116\n121 104\n121 115\n8\n8 234\n9 244\n10 226\n11 227\n12 240\n13 237\n14 206\n15 227"
    expected_output = "256\n255\n250\n247\n255\n259\n223\n253"
    run_pie_test_case("../p02648.py", input_content, expected_output)


def test_problem_p02648_2():
    input_content = "3\n1 2\n2 3\n3 4\n3\n1 1\n2 5\n3 5"
    expected_output = "0\n3\n3"
    run_pie_test_case("../p02648.py", input_content, expected_output)
