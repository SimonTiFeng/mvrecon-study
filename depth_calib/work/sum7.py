import numpy as np
B=[("courtyard",0.9524,28,0.0816,0.0210,27),("facade",0.9486,55,0.1209,0.0454,54),
   ("delivery_area",0.9379,33,0.0616,0.0346,28),("electro",0.9536,31,0.1035,0.0397,25),
   ("office",0.9687,16,0.0406,0.0269,13),("playground",0.9663,24,0.1872,0.1170,21),
   ("terrace",0.9735,13,0.0408,0.0272,9)]
print(f"{'scene':<15}{'const':>8}{'n':>4}{'raw':>9}{'cal':>9}{'wins':>8}{'delta':>9}")
tot=sum(b[2] for b in B); wn=sum(b[5] for b in B)
for s,c,n,r,k,w in B:
    print(f"{s:<15}{c:>8.4f}{n:>4}{r:>9.4f}{k:>9.4f}{w:>5}/{n:<3}{k-r:>+9.4f}")
rw=np.average([b[3] for b in B],weights=[b[2] for b in B])
cl=np.average([b[4] for b in B],weights=[b[2] for b in B])
cs=[b[1] for b in B]
print(f"\nimages={tot}  weighted raw {rw:.4f} -> cal {cl:.4f}   wins {wn}/{tot} = {wn/tot*100:.1f}%")
print(f"constants: median {np.median(cs):.4f}  sd {np.std(cs):.4f}  range [{min(cs):.4f},{max(cs):.4f}]")
print("published DA3-metric ETH3D = 0.104")
# zero-param pooled from each batch
Z=[("courtyard,facade",83,0.1094,0.0585,67),("delivery_area,electro",64,0.0928,0.0595,44),
   ("office,playground,terrace",53,0.0968,0.0661,30)]
wr=np.average([z[2] for z in Z],weights=[z[1] for z in Z])
zc=np.average([z[3] for z in Z],weights=[z[1] for z in Z])
zw=sum(z[4] for z in Z); zt=sum(z[1] for z in Z)
print(f"ZERO-PARAM weighted {wr:.4f} -> {zc:.4f}  wins {zw}/{zt} = {zw/zt*100:.1f}%")
