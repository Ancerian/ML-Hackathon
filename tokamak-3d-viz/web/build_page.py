#!/usr/bin/env python3
"""Збирає самодостатню сторінку переглядача: index.html + вбудовані jet1975.glb, jet1983.glb (base64)
і jet_meta.json → dist/index.html (хостинг артефактів не віддає .glb як окремі файли)."""
import base64, json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
tpl = (HERE / "index.html").read_text(encoding="utf-8")
blocks = []
for name in ("jet1975", "jet1983"):
    b64 = base64.b64encode((HERE / f"{name}.glb").read_bytes()).decode("ascii")
    blocks.append(f'<script type="text/plain" id="glb-{name}">{b64}</script>')
meta = json.dumps(json.loads((HERE / "jet_meta.json").read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))
meta = meta.replace("</", "<\\/")
blocks.append(f'<script type="application/json" id="jet-meta">{meta}</script>')
marker = '<script src="https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js"></script>'
assert marker in tpl
out = tpl.replace(marker, "\n".join(blocks) + "\n" + marker, 1)
dist = HERE / "dist"; dist.mkdir(exist_ok=True)
(dist / "index.html").write_text(out, encoding="utf-8")
print(f"dist/index.html: {len(out.encode('utf-8'))/1e6:.2f} MB")
