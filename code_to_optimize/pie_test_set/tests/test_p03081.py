from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03081_0():
    input_content = "3 4\nABC\nA L\nB L\nB R\nA R"
    expected_output = "2"
    run_pie_test_case("../p03081.py", input_content, expected_output)


def test_problem_p03081_1():
    input_content = "10 15\nSNCZWRCEWB\nB R\nR R\nE R\nW R\nZ L\nS R\nQ L\nW L\nB R\nC L\nA L\nN L\nE R\nZ L\nS L"
    expected_output = "3"
    run_pie_test_case("../p03081.py", input_content, expected_output)


def test_problem_p03081_2():
    input_content = "3 4\nABC\nA L\nB L\nB R\nA R"
    expected_output = "2"
    run_pie_test_case("../p03081.py", input_content, expected_output)


def test_problem_p03081_3():
    input_content = "8 3\nAABCBDBA\nA L\nB R\nA R"
    expected_output = "5"
    run_pie_test_case("../p03081.py", input_content, expected_output)
