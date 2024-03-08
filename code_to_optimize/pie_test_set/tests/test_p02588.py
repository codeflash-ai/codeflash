from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02588_0():
    input_content = "5\n7.5\n2.4\n17.000000001\n17\n16.000000000"
    expected_output = "3"
    run_pie_test_case("../p02588.py", input_content, expected_output)


def test_problem_p02588_1():
    input_content = "5\n7.5\n2.4\n17.000000001\n17\n16.000000000"
    expected_output = "3"
    run_pie_test_case("../p02588.py", input_content, expected_output)


def test_problem_p02588_2():
    input_content = (
        "11\n0.9\n1\n1\n1.25\n2.30000\n5\n70\n0.000000001\n9999.999999999\n0.999999999\n1.000000001"
    )
    expected_output = "8"
    run_pie_test_case("../p02588.py", input_content, expected_output)
