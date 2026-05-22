from web3 import Web3
from app.config import settings

def get_web3():
    return Web3(Web3.HTTPProvider(settings.ALCHEMY_RPC_URL, request_kwargs={"timeout": 5}))

def validate_address(address: str) -> dict:
    return {
        "address": address,
        "is_valid_checksum": Web3.is_checksum_address(address),
        "is_valid_address": Web3.is_address(address)
    }

def resolve_ens(name: str) -> dict:
    try:
        w3 = get_web3()
        address = w3.ens.address(name)
        return {
            "ens": name,
            "resolved_address": address,
            "valid": address is not None
        }
    except Exception as e:
        return {"ens": name, "resolved_address": None, "valid": False, "error": str(e)}

def enrich_signals(signals: dict) -> dict:
    return {
        "eth_addresses": [validate_address(a) for a in signals["eth_addresses"]],
        "ens_names": [resolve_ens(e) for e in signals["ens_names"]],
        "tickers": signals["tickers"]   # no on-chain resolution needed
    }
