from web3 import Web3
w = Web3()
print(type(w.ens))
print(hasattr(w.ens, 'address'))
print(hasattr(w.ens, 'resolve'))
