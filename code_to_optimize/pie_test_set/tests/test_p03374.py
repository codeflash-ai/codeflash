from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03374_0():
    input_content = "3 20\n2 80\n9 120\n16 1"
    expected_output = "191"
    run_pie_test_case("../p03374.py", input_content, expected_output)


def test_problem_p03374_1():
    input_content = "15 10000000000\n400000000 1000000000\n800000000 1000000000\n1900000000 1000000000\n2400000000 1000000000\n2900000000 1000000000\n3300000000 1000000000\n3700000000 1000000000\n3800000000 1000000000\n4000000000 1000000000\n4100000000 1000000000\n5200000000 1000000000\n6600000000 1000000000\n8000000000 1000000000\n9300000000 1000000000\n9700000000 1000000000"
    expected_output = "6500000000"
    run_pie_test_case("../p03374.py", input_content, expected_output)


def test_problem_p03374_2():
    input_content = "3 20\n2 80\n9 120\n16 1"
    expected_output = "191"
    run_pie_test_case("../p03374.py", input_content, expected_output)


def test_problem_p03374_3():
    input_content = "3 20\n2 80\n9 1\n16 120"
    expected_output = "192"
    run_pie_test_case("../p03374.py", input_content, expected_output)


def test_problem_p03374_4():
    input_content = "1 100000000000000\n50000000000000 1"
    expected_output = "0"
    run_pie_test_case("../p03374.py", input_content, expected_output)
