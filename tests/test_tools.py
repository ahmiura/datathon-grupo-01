from src.agent.tools import calculator


def test_calculator_valid_simple():
    res = calculator("2 + 3 * 4")
    assert res == "14"


def test_calculator_valid_float():
    res = calculator("10 / 4")
    # 10/4 = 2.5
    assert res == "2.5"


def test_calculator_invalid_chars():
    res = calculator("import os")
    assert "Erro" in res


def test_calculator_invalid_operator():
    # bitwise operator should be rejected by allowed chars check
    res = calculator("2 & 3")
    assert "Erro" in res


def test_calculator_mod_and_floordiv():
    res1 = calculator("10 % 3")
    assert res1 == "1"

    res2 = calculator("10 // 3")
    assert res2 == "3"


def test_calculator_decimal_precision():
    # check decimal division precision
    res = calculator("1 / 3")
    # should be a decimal string with reasonable precision
    assert res.startswith("0.")
