import json, sys, datetime, statistics as S
H = sys.argv[1]
prov = [json.loads(l) for l in open(H + "/history/provenance.jsonl")]
print("provenance rows", len(prov), [(p.get("event"), p.get("tick"), p.get("server_version")) for p in prov])
b = prov[0]; print("boot", datetime.datetime.utcfromtimestamp(b["started_at_unix"]).isoformat(), "UTC; arch", b.get("arch"))
cfg = b["cfg"]
for k in ("arm_shares", "base_food_drain", "rest_drain_mult", "collision_cost", "max_nxers", "min_alive", "brain_step_every", "autopilot", "start_food", "steal_cooldown_ticks", "m_selection_drain_relief", "m_selection_forage_bonus"):
    print("  cfg", k, cfg.get(k))
ts = [json.loads(l) for l in open(H + "/history/timeseries.jsonl")]
print("timeseries rows", len(ts), "first tick", ts[0]["tick"], "last", ts[-1]["tick"], "uptime days", round(ts[-1]["uptime_s"] / 86400, 3))
tps = [(ts[i]["tick"] - ts[i-1]["tick"]) / max(1e-9, ts[i]["uptime_s"] - ts[i-1]["uptime_s"]) for i in range(1, len(ts))]
print("TPS median", round(S.median(tps), 2), "overall", round((ts[-1]["tick"] - ts[0]["tick"]) / (ts[-1]["uptime_s"] - ts[0]["uptime_s"]), 2))
ob = [json.loads(l) for l in open(H + "/history/obituaries.jsonl")]
ids = set(o["id"] for o in ob); print("obituary rows", len(ob), "unique ids", len(ids))
lin = [json.loads(l) for l in open(H + "/history/lineage.jsonl")]
nas = [json.loads(l) for l in open(H + "/history/nas_trials.jsonl")]
print("births by mating (lineage)", len(lin), "NAS explorer trials logged (dead)", len(nas))
snap = json.load(open(H + "/snapshot.prev.json"))
alive = [x for x in snap["nxers"] if x.get("alive", True)]
print("snapshot tick", snap["tick"], "alive", len(alive), "next id", snap["next_nxer_id"], "nas_trial_seq", snap["nas_trial_seq"])
from collections import Counter
print("deaths by arm", Counter(o.get("arm") for o in ob)); print("alive by arm", Counter(x.get("arm") for x in alive))
print("causes", Counter(o.get("cause") for o in ob))
print("explorer share of all NxErs ever", round(sum(1 for o in ob if o.get("explorer")) / len(ob), 3), "alive explorers", sum(1 for x in alive if x.get("explorer")))
