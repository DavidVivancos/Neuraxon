"""Show a try: a GIF of its frames (needs Pillow) or the board in the terminal."""

# the ARC-AGI-3 colours 0..15
PALETTE = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#000000', '#E53AA3', '#FF7BCC', '#F93C31', '#1E93FF',
           '#88D8F1', '#FFDC00', '#FF851B', '#921231', '#4FCC30', '#A356D6']
_RGB = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in PALETTE]


def gif(events, path, scale=6, ms=250, brain_only=False):
    """write a GIF of a try's frames (play(..., frames=True)); a white bar under the board marks the brain's moves"""
    from PIL import Image
    imgs = []
    for e in events:
        if not e.get('frame') or (brain_only and e['by'] != 'brain'):
            continue
        g = e['frame'][0]
        h, w = len(g), len(g[0]) if g else 0
        if not h or not w:
            continue
        im = Image.new('RGB', (w, h + 2))
        px = im.load()
        for y in range(h):
            for x in range(w):
                px[x, y] = _RGB[g[y][x] & 15]
        bar = (255, 255, 255) if e['by'] == 'brain' else (90, 90, 90)
        for x in range(w):
            px[x, h] = px[x, h + 1] = bar
        imgs.append(im.resize((w * scale, (h + 2) * scale), Image.NEAREST))
    if not imgs:
        raise ValueError('no frames: play with frames=True')
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0)
    return path


def board(frame):
    """the board as coloured blocks for a terminal (ANSI 24-bit colour)"""
    out = []
    for row in frame:
        out.append(''.join(f'\x1b[48;2;{_RGB[c & 15][0]};{_RGB[c & 15][1]};{_RGB[c & 15][2]}m  ' for c in row) + '\x1b[0m')
    return '\n'.join(out)
