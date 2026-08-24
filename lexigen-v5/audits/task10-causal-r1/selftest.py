from __future__ import annotations
import random,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
LEXIGEN=HERE.parents[1]
TASK=LEXIGEN/'tasks'/'10-vertex-cover'
sys.path.insert(0,str(TASK));sys.path.insert(0,str(HERE))
from candidates import reference_exact
from audit_algorithms import reproduced_bfr,color_bound_clique_cover


def valid(p,cover):
    n=len(p);s=set(cover)
    if len(s)!=len(cover) or any(type(x) is not int or x<0 or x>=n for x in cover):return False
    return all(not p[i][j] or i in s or j in s for i in range(n) for j in range(i+1,n))


def er(n,p,seed):
    r=random.Random(seed);a=[[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if r.random()<p:a[i][j]=a[j][i]=1
    return a


def cases():
    out=[]
    for n in range(1,15):
        for k,p in enumerate((0.0,0.1,0.3,0.5,0.7,0.9,1.0)):
            out.append((f'er_n{n}_p{p}',er(n,p,10000+n*100+k)))
    for n in (2,3,5,8,13):
        star=[[0]*n for _ in range(n)]
        for j in range(1,n):star[0][j]=star[j][0]=1
        out.append((f'star_{n}',star))
        cyc=[[0]*n for _ in range(n)]
        if n>2:
            for i in range(n):cyc[i][(i+1)%n]=cyc[(i+1)%n][i]=1
        out.append((f'cycle_{n}',cyc))
    return out


def main():
    total=0
    for name,p in cases():
        ref=reference_exact(p);r=reproduced_bfr(p);c=color_bound_clique_cover(p)
        if not valid(p,ref) or not valid(p,r) or not valid(p,c):raise RuntimeError(f'invalid cover {name}')
        sizes=(len(ref),len(r),len(c))
        if len(set(sizes))!=1:raise RuntimeError(f'optimum mismatch {name}: {sizes}')
        total+=1
    print({'status':'passed','cases':total})
if __name__=='__main__':main()
