# Can an IDEAL sign-steering reflex eat, given one decision per 8 ticks with replay?
import random, math, sys
CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 8
R = CELL  # 3x3 buckets ~ +-CELL..2*CELL; use Chebyshev window of 1.5*CELL
W = int(1.5 * CELL)
DIRS=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
def sg(v): return 1 if v>0 else (-1 if v<0 else 0)
def run(mode, K=8, dens=0.05, T=30000, seed=1):
    rnd = random.Random(seed); N = 200
    food = {}
    while len(food) < int(N*N*dens):
        food[(rnd.randrange(N), rnd.randrange(N))] = 25
    x = y = 100; out = (0, 0, 0); last_eat = -99; bites = 0; hold = False
    for t in range(T):
        if t % K == 0 or mode == "per_tick":
            best = None; bd = 1e9
            for fx in range(x-W, x+W+1):
                for fy in range(y-W, y+W+1):
                    if (fx % N, fy % N) in food:
                        d = (fx-x)**2 + (fy-y)**2
                        if d < bd: bd = d; best = (fx, fy)
            if mode.startswith("random"):
                h = DIRS[rnd.randrange(8)]; out = (h[0], h[1], 0)
            elif best is None:
                h = DIRS[rnd.randrange(8)]; out = (h[0], h[1], 0)
            else:
                dx, dy = sg(best[0]-x), sg(best[1]-y)
                out = (dx, dy, 1 if (dx, dy) == (0, 0) else 0)   # rest on food
            hold = False
        mx, my, rest = out
        if hold or rest: mx = my = 0
        if mode in ("one_step", "random_one") and t % K != 0: mx = my = 0
        x += mx; y += my
        k = (x % N, y % N)
        if k in food and t - last_eat >= 20:
            food[k] -= 1; bites += 1; last_eat = t
            if food[k] <= 0:
                del food[k]
                while True:
                    nk = (rnd.randrange(N), rnd.randrange(N))
                    if nk not in food: food[nk] = 25; break
        if mode in ("stop_on_food", "random_stop") and k in food:
            hold = True       # consummatory stop: replay halts on a food cell
    return bites / T * 1000
for dens in (0.02, 0.05):
    ms=("per_tick","replay8","stop_on_food","one_step","random","random_stop","random_one")
    r = {m: sum(run(m, dens=dens, seed=s) for s in range(3))/3 for m in ms}
    print("density %.2f: " % dens + " | ".join("%s %.1f" % (m, r[m]) for m in ms))
