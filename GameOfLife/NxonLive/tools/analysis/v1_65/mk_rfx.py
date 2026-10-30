import sys, os, json, random
ROOT = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, ROOT); os.environ["NEURAXON_HEADLESS"]="1"
from server import np_fallback; np_fallback.install()
import architecture
architecture.load_architecture(os.path.join(ROOT,"architecture_files","nas_best.json"), verbose=False)
from neuraxon.multisphere import build_brain
from server.engine import make_params
for k,(g,arm) in enumerate([({"rfx_xx":1,"rfx_xy":0,"rfx_yx":0,"rfx_yy":1,"rfx_rest":0,"rfx_gain":1.5},"reflex"),
                            ({"rfx_xx":-1,"rfx_xy":0,"rfx_yx":0,"rfx_yy":-1,"rfx_rest":0,"rfx_gain":1.5},"reflex"),
                            ({"rfx_xx":1,"rfx_xy":0,"rfx_yx":0,"rfx_yy":1,"rfx_rest":0,"rfx_gain":1.5},"brain")]):
    random.seed(k)
    b = build_brain(make_params(None)).to_dict()
    json.dump({"name":"t%d"%k,"brain":b,"params":None,"reflex":g,"arm":arm}, open(os.path.join(OUT,"t%d.json"%k),"w"))
print("made")
