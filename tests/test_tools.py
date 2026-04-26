from src.agent.tools import calculator


def test_calculator_valid_simple():
    res = calculator.invoke("2 + 3 * 4")
    assert res == "14.0"


def test_calculator_valid_float():
    res = calculator.invoke("10 / 4")
    # 10/4 = 2.5
    assert res == "2.5"


def test_calculator_invalid_chars():
    res = calculator.invoke("import os")
    assert "Erro" in res


def test_calculator_invalid_operator():
    # bitwise operator should be rejected by allowed chars check
    res = calculator.invoke("2 & 3")
    assert "Erro" in res


def test_calculator_mod_and_floordiv():
    res1 = calculator.invoke("10 % 3")
    assert res1 == "1.0"

    res2 = calculator.invoke("10 // 3")
    assert res2 == "3.0"


def test_calculator_decimal_precision():
    # check decimal division precision
    res = calculator.invoke("1 / 3")
    # should be a decimal string with reasonable precision
    assert res.startswith("0.")
