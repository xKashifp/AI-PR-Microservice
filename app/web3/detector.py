import re

ETH_ADDRESS_RE = re.compile(r'\b0x[a-fA-F0-9]{40}\b')
ENS_RE = re.compile(r'\b[\w-]+\.eth\b')
TICKER_RE = re.compile(r'\$[A-Z]{2,10}\b')

def detect_web3_signals(text: str) -> dict:
    return {
        "eth_addresses": ETH_ADDRESS_RE.findall(text),
        "ens_names": ENS_RE.findall(text),
        "tickers": TICKER_RE.findall(text)
    }
