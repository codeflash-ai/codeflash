from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03695_0():
    input_content = "4\n2100 2500 2700 2700"
    expected_output = "2 2"
    run_pie_test_case("../p03695.py", input_content, expected_output)


def test_problem_p03695_1():
    input_content = "4\n2100 2500 2700 2700"
    expected_output = "2 2"
    run_pie_test_case("../p03695.py", input_content, expected_output)


def test_problem_p03695_2():
    input_content = (
        "20\n800 810 820 830 840 850 860 870 880 890 900 910 920 930 940 950 960 970 980 990"
    )
    expected_output = "1 1"
    run_pie_test_case("../p03695.py", input_content, expected_output)


def test_problem_p03695_3():
    input_content = "5\n1100 1900 2800 3200 3200"
    expected_output = "3 5"
    run_pie_test_case("../p03695.py", input_content, expected_output)
