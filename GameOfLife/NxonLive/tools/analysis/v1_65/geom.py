# Geometry check: a sense-blind persistent walker, food field, sensing every 8 ticks.
import random, math
random.seed(1)
N=200; dens=0.03
food=set((random.randrange(N),random.randrange(N)) for _ in range(int(N*N*dens)))
DIRS=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
def nearest(x,y,R=12):
    best=None;bd=1e9
    for fx in range(x-R,x+R+1):
        for fy in range(y-R,y+R+1):
            if (fx%N,fy%N) in food:
                d=(fx-x)**2+(fy-y)**2
                if d<bd and d>0: bd=d;best=(fx,fy)
    return best
def sg(v): return 1 if v>0 else (-1 if v<0 else 0)
def cosv(dx,dy,ox,oy):
    sx,sy=sg(ox),sg(oy)
    if not (sx or sy): return None
    return (dx*sx+dy*sy)/math.sqrt((dx*dx+dy*dy)*(sx*sx+sy*sy))
for turn_p in (0.0, 0.2, 1.0):
    x=y=100; h=0; stale=[]; first=[]; fresh_all=[]
    for t in range(40000):
        if t%8==0:
            tgt=nearest(x,y)
            if random.random()<turn_p: h=random.randrange(8)
        dx,dy=DIRS[h]
        if tgt:
            c=cosv(dx,dy,tgt[0]-x,tgt[1]-y)
            if c is not None:
                stale.append(c)
                if t%8==0: first.append(c)
        nt=nearest(x,y)
        if nt:
            c2=cosv(dx,dy,nt[0]-x,nt[1]-y)
            if c2 is not None: fresh_all.append(c2)
        x+=dx;y+=dy
        if (x%N,y%N) in food:
            food.discard((x%N,y%N)); food.add((random.randrange(N),random.randrange(N)))
    m=lambda v: sum(v)/len(v)
    print("turn prob per decision %.1f: stale-target cosine %+.3f | first step after sensing %+.3f | re-sensed every step %+.3f" % (turn_p, m(stale), m(first), m(fresh_all)))
