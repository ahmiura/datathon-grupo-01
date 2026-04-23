import pytest

from security.guardrails import check_output


def test_mask_email():
    ok, out = check_output("Meu email é teste.user@example.com")
    assert not ok
    assert "[EMAIL:***@***]" in out


def test_mask_cnpj():
    ok, out = check_output("CNPJ: 12.345.678/0001-95")
    assert not ok
    assert "**.***.***/****-**" in out


def test_mask_cpf():
    ok, out = check_output("CPF 123.456.789-00 detectado")
    assert not ok
    assert "***.***.***-**" in out


def test_mask_bank_account():
    ok, out = check_output("Minha conta 123456789012")
    assert not ok
    assert "[BANK_ACCOUNT:***]" in out


def test_no_pii():
    ok, out = check_output("Qual o preço da ação?")
    assert ok
    assert out == "Qual o preço da ação?"
