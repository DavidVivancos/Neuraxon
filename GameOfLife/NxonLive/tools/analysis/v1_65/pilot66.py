"""Small-world pilot of v1.66 dynamics: per-arm executed chemotaxis, bites, survival, reflex gene drift."""
import os, sys, json, random, time
ROOT, SEED, TICKS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, ROOT); os.environ["NEURAXON_HEADLESS"] = "1"
from server import np_fallback; np_fallback.install()
import architecture
architecture.load_architecture(os.path.join(ROOT, "architecture_files", "nas_best.json"), verbose=False)
from server.engine import Engine
from server.names import NameAllocator
random.seed(SEED)
cfg = json.load(open(os.path.join(ROOT, "world_config.json")))
cfg.update({"engine_workers": 1, "brain_builders": 1, "reserved_builder_cores": 0,
            "world_size": 110, "earth_map": False, "starting_nxers": 100, "max_nxers": 140,
            "min_alive": 70, "max_food": 450, "history_enabled": False,
            "nas_explore_interval_ticks": 150})
class H:
    def __init__(s): s.obits=[]; s.lin=[]
    def obituary(s, r): s.obits.append(r)
    def lineage(s, r): s.lin.append(r)
    def write(s, *a): pass
    def sample(s, r): pass
    def provenance(s, r): pass
    def event(s, r): pass
eng = Engine(cfg, NameAllocator(0)); h = H(); eng.history = h
snaps = []; t0 = time.time()
for t in range(TICKS):
    eng.pool.drain_loads(); eng.step()
    if (t + 1) % 1000 == 0:
        live = []
        for x in eng.nxers.values():
            if not x.alive: continue
            b = Engine.behav(x)
            live.append({"arm": x.arm, "age": eng.tick - x.born_tick, "food_found": x.stats.food_found, "food_taken": x.stats.food_taken,
                         "reflex": getattr(x, "reflex", None), "cx": b.get("chemo_exec"),
                         "cxn": b.get("chemo_exec_n"), "appr": b.get("approach_frac"),
                         "rest": b.get("out_rest"), "lock": b.get("lock_frac"),
                         "e_eat": b.get("e_eat"), "e_stin": b.get("e_stin"), "e_drain": b.get("e_drain"), "e_coll": b.get("e_coll"), "e_stout": b.get("e_stout"), "motor": b.get("motor_frac"), "chemo": b.get("chemotaxis"), "chemo_n": b.get("chemo_n")})
        snaps.append({"tick": t + 1, "alive": len(live), "live": live, "wall": round(time.time() - t0)})
        json.dump({"snaps": snaps, "obits": h.obits, "lineage": h.lin}, open(OUT, "w"))
        print("tick", t + 1, "alive", len(live), "deaths", len(h.obits), "wall", round(time.time() - t0), flush=True)
