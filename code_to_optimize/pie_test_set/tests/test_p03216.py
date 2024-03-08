from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03216_0():
    input_content = "18\nDWANGOMEDIACLUSTER\n1\n18"
    expected_output = "1"
    run_pie_test_case("../p03216.py", input_content, expected_output)


def test_problem_p03216_1():
    input_content = "54\nDIALUPWIDEAREANETWORKGAMINGOPERATIONCORPORATIONLIMITED\n3\n20 30 40"
    expected_output = "0\n1\n2"
    run_pie_test_case("../p03216.py", input_content, expected_output)


def test_problem_p03216_2():
    input_content = "18\nDWANGOMEDIACLUSTER\n1\n18"
    expected_output = "1"
    run_pie_test_case("../p03216.py", input_content, expected_output)


def test_problem_p03216_3():
    input_content = "18\nDDDDDDMMMMMCCCCCCC\n1\n18"
    expected_output = "210"
    run_pie_test_case("../p03216.py", input_content, expected_output)


def test_problem_p03216_4():
    input_content = "30\nDMCDMCDMCDMCDMCDMCDMCDMCDMCDMC\n4\n5 10 15 20"
    expected_output = "10\n52\n110\n140"
    run_pie_test_case("../p03216.py", input_content, expected_output)
