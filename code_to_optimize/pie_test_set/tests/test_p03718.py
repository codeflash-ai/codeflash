from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03718_0():
    input_content = "3 3\nS.o\n.o.\no.T"
    expected_output = "2"
    run_pie_test_case("../p03718.py", input_content, expected_output)


def test_problem_p03718_1():
    input_content = "4 3\n.S.\n.o.\n.o.\n.T."
    expected_output = "-1"
    run_pie_test_case("../p03718.py", input_content, expected_output)


def test_problem_p03718_2():
    input_content = "3 4\nS...\n.oo.\n...T"
    expected_output = "0"
    run_pie_test_case("../p03718.py", input_content, expected_output)


def test_problem_p03718_3():
    input_content = "3 3\nS.o\n.o.\no.T"
    expected_output = "2"
    run_pie_test_case("../p03718.py", input_content, expected_output)


def test_problem_p03718_4():
    input_content = "10 10\n.o...o..o.\n....o.....\n....oo.oo.\n..oooo..o.\n....oo....\n..o..o....\no..o....So\no....T....\n....o.....\n........oo"
    expected_output = "5"
    run_pie_test_case("../p03718.py", input_content, expected_output)
