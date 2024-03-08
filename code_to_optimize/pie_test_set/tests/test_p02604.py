from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02604_0():
    input_content = "3\n1 2 300\n3 3 600\n1 4 800"
    expected_output = "2900\n900\n0\n0"
    run_pie_test_case("../p02604.py", input_content, expected_output)


def test_problem_p02604_1():
    input_content = "8\n2 2 286017\n3 1 262355\n2 -2 213815\n1 -3 224435\n-2 -2 136860\n-3 -1 239338\n-2 2 217647\n-1 3 141903"
    expected_output = "2576709\n1569381\n868031\n605676\n366338\n141903\n0\n0\n0"
    run_pie_test_case("../p02604.py", input_content, expected_output)


def test_problem_p02604_2():
    input_content = "5\n3 5 400\n5 3 700\n5 5 1000\n5 7 700\n7 5 400"
    expected_output = "13800\n1600\n0\n0\n0\n0"
    run_pie_test_case("../p02604.py", input_content, expected_output)


def test_problem_p02604_3():
    input_content = "6\n2 5 1000\n5 2 1100\n5 5 1700\n-2 -5 900\n-5 -2 600\n-5 -5 2200"
    expected_output = "26700\n13900\n3200\n1200\n0\n0\n0"
    run_pie_test_case("../p02604.py", input_content, expected_output)


def test_problem_p02604_4():
    input_content = "3\n1 2 300\n3 3 600\n1 4 800"
    expected_output = "2900\n900\n0\n0"
    run_pie_test_case("../p02604.py", input_content, expected_output)
