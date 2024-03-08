from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02264_0():
    input_content = "5 100\np1 150\np2 80\np3 200\np4 350\np5 20"
    expected_output = "p2 180\np5 400\np1 450\np3 550\np4 800"
    run_pie_test_case("../p02264.py", input_content, expected_output)


def test_problem_p02264_1():
    input_content = "5 100\np1 150\np2 80\np3 200\np4 350\np5 20"
    expected_output = "p2 180\np5 400\np1 450\np3 550\np4 800"
    run_pie_test_case("../p02264.py", input_content, expected_output)
