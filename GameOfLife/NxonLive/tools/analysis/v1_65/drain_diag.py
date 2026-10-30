# Where does metabolism go? per-arm mean atrophy factor and share of idle ticks
import os, sys, json, random
ROOT = sys.argv[1]; TICKS = int(sys.argv[2])
sys.path.insert(0, ROOT); os.environ["NEURAXON_HEADLESS"] = "1"
from server import np_fallback; np_fallback.install()
import architecture
architecture.load_architecture(os.path.join(ROOT, "architecture_files", "nas_best.json"), verbose=False)
from server.engine import Engine
from server.names import NameAllocator
random.seed(5)
cfg = json.load(open(os.path.join(ROOT, "world_config.json")))
cfg.update({"engine_workers": 1, "brain_builders": 1, "reserved_builder_cores": 0, "world_size": 90, "earth_map": False,
            "starting_nxers": 60, "max_nxers": 90, "min_alive": 40, "max_food": 300, "history_enabled": False})
eng = Engine(cfg, NameAllocator(0))
acc = {}
for t in range(TICKS):
    eng.pool.drain_loads(); eng.step()
    if t < 100: continue
    for x in eng.nxers.values():
        if not x.alive: continue
        a = acc.setdefault(x.arm, {"n": 0, "idle0": 0, "atr": 0.0, "idle_long": 0, "drain": 0.0, "age": 0})
        idle = eng.tick - x.last_move_tick
        atr = min(eng.bio_max_atrophy, 1.0 + eng.bio_metab_ramp * idle * eng.dt)
        a["n"] += 1; a["idle0"] += idle == 0; a["atr"] += atr; a["idle_long"] += idle > 64
print("ramp per s", round(eng.bio_metab_ramp, 3), "dt", eng.dt, "base", eng.bio_base_drain)
for arm, a in sorted(acc.items()):
    print("%-11s samples %6d moved-this-tick %.2f idle>64 ticks %.2f mean atrophy(if not resting) %.2f" % (arm, a["n"], a["idle0"]/a["n"], a["idle_long"]/a["n"], a["atr"]/a["n"]))
for arm in sorted(acc):
    v = [x for x in eng.nxers.values() if x.alive and x.arm == arm]
    T = sum(eng.tick - x.born_tick for x in v) or 1
    print("%-11s e_drain/1kt %.2f  relief/1kt %.2f  rest share %.2f" % (arm, sum(x.e_drain for x in v)/T*1000, sum(x.e_relief for x in v)/T*1000,
          sum(x._rest1 for x in v)/max(1, sum(x._dec_n for x in v))))
