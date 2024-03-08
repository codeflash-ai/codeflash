from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03152_0():
    input_content = "2 2\n4 3\n3 4"
    expected_output = "2"
    run_pie_test_case("../p03152.py", input_content, expected_output)


def test_problem_p03152_1():
    input_content = "3 3\n5 9 7\n3 6 9"
    expected_output = "0"
    run_pie_test_case("../p03152.py", input_content, expected_output)


def test_problem_p03152_2():
    input_content = "14 13\n158 167 181 147 178 151 179 182 176 169 180 129 175 168\n181 150 178 179 167 180 176 169 182 177 175 159 173"
    expected_output = "343772227"
    run_pie_test_case("../p03152.py", input_content, expected_output)


def test_problem_p03152_3():
    input_content = "2 2\n4 4\n4 4"
    expected_output = "0"
    run_pie_test_case("../p03152.py", input_content, expected_output)


def test_problem_p03152_4():
    input_content = "2 2\n4 3\n3 4"
    expected_output = "2"
    run_pie_test_case("../p03152.py", input_content, expected_output)
