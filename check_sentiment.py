import os
os.environ["OMP_NUM_THREADS"] = "1"
from app.nlp.sentiment import analyze

res_pos = analyze("This is wonderful news!")
res_neg = analyze("This is a complete disaster.")
print("POS:", res_pos)
print("NEG:", res_neg)
