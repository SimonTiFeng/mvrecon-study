import numpy as np
# per-batch results (held-out scene, n, raw, cal, wins)
B=[("courtyard",38,0.0965,0.0213,37),("facade",73,0.1193,0.0429,72),
   ("delivery_area",44,0.0582,0.0450,27),("electro",42,0.0927,0.1003,25),
   ("office",21,0.0454,0.0331,16),("playground",32,0.1705,0.1143,28),
   ("terrace",17,0.0435,0.0298,14)]
print(f"{'scene':<15}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>8}{'delta':>9}")
tot=0
for s,n,r,c,w in B:
    print(f"{s:<15}{n:>4}{r:>9.4f}{c:>9.4f}{w:>5}/{n:<3}{c-r:>+9.4f}")
    tot+=n
print(f"\ntotal images {tot}")
# approximate pooled: weight medians by n for a rough overall picture
rw=np.average([b[2] for b in B],weights=[b[1] for b in B])
cl=np.average([b[3] for b in B],weights=[b[1] for b in B])
wn=sum(b[4] for b in B)
print(f"weighted-mean raw {rw:.4f} -> cal {cl:.4f}   wins {wn}/{tot} = {wn/tot*100:.1f}%")
print("\nconstants used per hold-out (from the other scenes in that batch):")
for s,n,r,c,w in B: pass
print("  courtyard 0.9522/0.9535, facade 0.9486, delivery_area 0.9270,")
print("  electro 0.9554, office 0.9901, playground 0.9716, terrace 0.9486")
print(f"  spread sd = {np.std([0.9522,0.9486,0.9270,0.9554,0.9901,0.9716,0.9486]):.4f}")
