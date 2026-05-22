import pytest
from app.web3.detector import detect_web3_signals
from unittest.mock import patch, MagicMock


def test_detect_eth_address():
    text = "Wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 received funds"
    signals = detect_web3_signals(text)
    assert len(signals["eth_addresses"]) == 1
    assert signals["eth_addresses"][0] == "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"


def test_detect_ens():
    signals = detect_web3_signals("Send to vitalik.eth for the DAO")
    assert "vitalik.eth" in signals["ens_names"]


def test_detect_ticker():
    signals = detect_web3_signals("$ETH and $BTC are up today")
    assert "$ETH" in signals["tickers"]
    assert "$BTC" in signals["tickers"]


def test_detect_no_signals():
    signals = detect_web3_signals("Regular news article about business growth.")
    assert signals["eth_addresses"] == []
    assert signals["ens_names"] == []
    assert signals["tickers"] == []


def test_detect_multiple_addresses():
    text = "Transfer from 0xAbCdEf1234567890AbCdEf1234567890AbCdEf12 to 0x1234567890AbCdEf1234567890AbCdEf12345678"
    signals = detect_web3_signals(text)
    assert len(signals["eth_addresses"]) == 2


def test_detect_multiple_tickers():
    signals = detect_web3_signals("Portfolio: $ETH $BTC $SOL $MATIC $AVAX all rallied")
    assert len(signals["tickers"]) == 5


def test_validate_address():
    from app.web3.resolver import validate_address
    result = validate_address("0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
    assert result["is_valid_address"] is True
    assert "is_valid_checksum" in result


def test_resolve_ens_fallback():
    """ENS resolution gracefully returns error dict on failure."""
    from app.web3.resolver import resolve_ens
    with patch("app.web3.resolver.get_web3") as mock_w3:
        mock_w3.return_value.ens.address.side_effect = Exception("RPC timeout")
        result = resolve_ens("vitalik.eth")
    assert result["ens"] == "vitalik.eth"
    assert result["valid"] is False
    assert "error" in result


def test_enrich_signals():
    from app.web3.resolver import enrich_signals
    signals = {
        "eth_addresses": ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],
        "ens_names": [],
        "tickers": ["$ETH"]
    }
    enriched = enrich_signals(signals)
    assert len(enriched["eth_addresses"]) == 1
    assert enriched["tickers"] == ["$ETH"]
