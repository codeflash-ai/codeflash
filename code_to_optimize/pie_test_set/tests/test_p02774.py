from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02774_0():
    input_content = "4 3\n3 3 -4 -2"
    expected_output = "-6"
    run_pie_test_case("../p02774.py", input_content, expected_output)


def test_problem_p02774_1():
    input_content = "10 40\n5 4 3 2 -1 0 0 0 0 0"
    expected_output = "6"
    run_pie_test_case("../p02774.py", input_content, expected_output)


def test_problem_p02774_2():
    input_content = "4 3\n3 3 -4 -2"
    expected_output = "-6"
    run_pie_test_case("../p02774.py", input_content, expected_output)


def test_problem_p02774_3():
    input_content = "30 413\n-170202098 -268409015 537203564 983211703 21608710 -443999067 -937727165 -97596546 -372334013 398994917 -972141167 798607104 -949068442 -959948616 37909651 0 886627544 -20098238 0 -948955241 0 -214720580 277222296 -18897162 834475626 0 -425610555 110117526 663621752 0"
    expected_output = "448283280358331064"
    run_pie_test_case("../p02774.py", input_content, expected_output)
