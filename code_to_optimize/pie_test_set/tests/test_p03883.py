from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03883_0():
    input_content = "4\n2 7\n2 5\n4 1\n7 5"
    expected_output = "22"
    run_pie_test_case("../p03883.py", input_content, expected_output)


def test_problem_p03883_1():
    input_content = "20\n97 2\n75 25\n82 84\n17 56\n32 2\n28 37\n57 39\n18 11\n79 6\n40 68\n68 16\n40 63\n93 49\n91 10\n55 68\n31 80\n57 18\n34 28\n76 55\n21 80"
    expected_output = "7337"
    run_pie_test_case("../p03883.py", input_content, expected_output)


def test_problem_p03883_2():
    input_content = "4\n2 7\n2 5\n4 1\n7 5"
    expected_output = "22"
    run_pie_test_case("../p03883.py", input_content, expected_output)
