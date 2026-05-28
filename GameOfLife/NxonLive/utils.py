# Neuraxon Game of Life v.4.92 utils (Research Version):(Multi - Neuraxon 2.0 Compliant) Internal version 184
# Based on the Papers:
#   "Neuraxon V2.0: A New Neural Growth & Computation Blueprint" by David Vivancos & Jose Sanchez
#   https://vivancos.com/ & https://josesanchezgarcia.com/ for Qubic Science https://qubic.org/
# https://www.researchgate.net/publication/400868863_Neuraxon_V20_A_New_Neural_Growth_Computation_Blueprint  (Neuraxon V2.0 )
# https://www.researchgate.net/publication/397331336_Neuraxon (V1) 
# Play the Lite Version of the Game of Life 5 at https://huggingface.co/spaces/DavidVivancos/NeuraxonLife
import os
import sys
import math
import random
import json
import pathlib
from datetime import datetime
from datetime import timezone
from typing import List, Tuple, Optional, Any, Dict

# Import constants for color generation
from config import RESERVED_COLORS

def _variate(val: float, variance: float = 0.2) -> float:
    """Helper to apply biological heterogeneity to parameters."""
    return val * random.uniform(1.0 - variance, 1.0 + variance)

def _clamp(v, a, b):
    """Clamps value v between a and b."""
    return max(a, min(b, v))

def _now_str(): 
    """Returns current UTC timestamp as string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def _safe_path(name: str) -> str: 
    """Resolves a filename to an absolute path in the current working directory."""
    return str((pathlib.Path(os.getcwd()) / name).resolve())

def _strip_leading_digits(name: str) -> str:
    """Removes leading digits from a string, used for generating names for offspring."""
    i = 0
    while i < len(name) and name[i].isdigit(): i += 1
    return name[i:] if i < len(name) else ""

### HOTFIX4: O(1) deterministic color generator using golden-angle HSV spacing.
### The old version was while-True with O(N) distance check — froze at 200+ colors.
_hf4_color_idx = [0]
def _rand_color(exclude=None) -> Tuple[int, int, int]:
    """Generates a unique color via golden-angle HSV. O(1), never loops."""
    _hf4_color_idx[0] += 1
    n = _hf4_color_idx[0]
    # Golden angle ≈ 137.508° gives maximum hue separation
    hue = (n * 137.508) % 360.0
    # Vary saturation and value to avoid similar colors at same hue
    sat = 0.55 + (((n * 23) % 40) / 100.0)   # 0.55–0.95
    val = 0.50 + (((n * 17) % 45) / 100.0)   # 0.50–0.95
    # HSV to RGB (inline, no import needed)
    h_i = int(hue / 60.0) % 6
    f = (hue / 60.0) - int(hue / 60.0)
    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)
    if h_i == 0:   r, g, b = val, t, p
    elif h_i == 1: r, g, b = q, val, p
    elif h_i == 2: r, g, b = p, val, t
    elif h_i == 3: r, g, b = p, q, val
    elif h_i == 4: r, g, b = t, p, val
    else:          r, g, b = val, p, q
    return (int(r * 255), int(g * 255), int(b * 255))

def _rot(x, y, a):
    """Rotates a 2D point (x, y) around the origin by angle 'a'."""
    ca, sa = math.cos(a), math.sin(a)
    return (x * ca - y * sa, x * sa + y * ca)

def _chunked(seq, n):
    """Yields successive n-sized chunks from a sequence."""
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def _is_headless() -> bool:
    """v161 (v4.69) — detect headless mode so file dialogs are bypassed.
    
    The file-save dialog in _pick_save_file calls tkinter's
    asksaveasfilename(), which on Windows shows a modal dialog that waits
    for a user click. In a headless NAS subprocess with no real display,
    THIS HANGS FOREVER — the dialog might be invisible but the call
    blocks waiting for input that never arrives. The whole subprocess
    sits stuck and the Pool callback never fires.
    
    Detection signals (any one is enough):
      - NEURAXON_HEADLESS=1  (explicit flag, set by NxonArchNAS workers)
      - SDL_VIDEODRIVER=dummy  (pygame's dummy-driver mode)
      - SDL_AUDIODRIVER=dummy  (pygame's dummy-audio mode)
      - DISPLAY unset on Linux  (no X server)
    
    When headless, _pick_save_file returns _safe_path(default_name)
    instead of opening a dialog.
    """
    if os.environ.get('NEURAXON_HEADLESS') == '1':
        return True
    if os.environ.get('SDL_VIDEODRIVER') == 'dummy':
        return True
    if os.environ.get('SDL_AUDIODRIVER') == 'dummy':
        return True
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return True
    return False


def _pick_save_file(default_name: str) -> Optional[str]:
    """Opens a native OS file dialog to choose a save location.
    
    v161 (v4.69) — in headless mode (NAS workers), the tkinter dialog
    would hang forever waiting for a user click. Bypass entirely and
    save to the cwd with the given filename. See _is_headless().
    """
    if _is_headless():
        # Save to cwd with the provided filename — no dialog, no hang
        return _safe_path(default_name)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=default_name, filetypes=[("JSON files", "*.json")])
        root.destroy()
        return path if path else None
    except Exception:
        return _safe_path(default_name)

def _pick_open_file() -> Optional[str]:
    """Opens a native OS file dialog to choose a file to open.
    
    v161 (v4.69) — in headless mode, return the most recent save file
    in cwd as a no-dialog fallback (matches the existing except-branch).
    """
    if _is_headless():
        cand = sorted([p for p in os.listdir(os.getcwd()) if p.endswith(".json") and p.startswith("nxer_")])
        return cand[-1] if cand else None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        root.destroy()
        return path if path else None
    except Exception:
        # A simple fallback for environments without a GUI.
        cand = sorted([p for p in os.listdir(os.getcwd()) if p.endswith(".json") and p.startswith("nxer_")])
        return cand[-1] if cand else None
def safe_json_save(data: Dict[str, Any], filepath: str, retries: int = 3) -> bool:
    """
    Saves a dictionary to JSON and verifies integrity by reading it back immediately.
    If verification fails, it retries.
    
    Args:
        data: The dictionary to save.
        filepath: The target file path.
        retries: Number of attempts before giving up.
        
    Returns:
        True if saved and verified successfully, False otherwise.
    """
    for i in range(retries):
        try:
            # 1. Write the data
            with open(filepath, "w") as f:
                json.dump(data, f) # Removed indent for performance/size on large saves, or add indent=2 if readability needed
            
            # 2. Read it back to verify integrity
            with open(filepath, "r") as f_check:
                json.load(f_check)
            
            # If we got here, the JSON is valid
            return True
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"[SAVE ERROR] Attempt {i+1}/{retries} failed for {filepath}: {e}")
            
    print(f"[SAVE FATAL] Could not save verified JSON to {filepath} after {retries} attempts.")
    
    # Emergency backup attempt (try saving with .corrupt suffix just in case it's a verify issue but data exists)
    try:
        emergency_path = filepath + ".unverified"
        with open(emergency_path, "w") as f:
            json.dump(data, f)
        print(f"[SAVE EMERGENCY] Saved unverified dump to {emergency_path}")
    except:
        pass
        
    return False
