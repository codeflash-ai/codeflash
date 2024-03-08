from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02863_0():
    input_content = "2 60\n10 10\n100 100"
    expected_output = "110"
    run_pie_test_case("../p02863.py", input_content, expected_output)


def test_problem_p02863_1():
    input_content = "3 60\n10 10\n10 20\n10 30"
    expected_output = "60"
    run_pie_test_case("../p02863.py", input_content, expected_output)


def test_problem_p02863_2():
    input_content = "10 100\n15 23\n20 18\n13 17\n24 12\n18 29\n19 27\n23 21\n18 20\n27 15\n22 25"
    expected_output = "145"
    run_pie_test_case("../p02863.py", input_content, expected_output)


def test_problem_p02863_3():
    input_content = "3 60\n30 10\n30 20\n30 30"
    expected_output = "50"
    run_pie_test_case("../p02863.py", input_content, expected_output)


def test_problem_p02863_4():
    input_content = "2 60\n10 10\n100 100"
    expected_output = "110"
    run_pie_test_case("../p02863.py", input_content, expected_output)
