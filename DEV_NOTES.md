# KVarN dev notes

Local maintenance notes for building and upgrading this repo. Not part of the
upstream vLLM docs; not user-facing.

## After bumping the vLLM base: reinstall `flashinfer-cubin`

`requirements/cuda.txt` pins both `flashinfer-python` and `flashinfer-cubin`,
but `setup.py` deliberately excludes `flashinfer-cubin` from `install_requires`
(it is not on PyPI since 0.6.14; upstream ships it only via
`https://flashinfer.ai/whl/`). Consequence: `uv pip install -e .` upgrades
`flashinfer-python` to the new pin and silently leaves a stale
`flashinfer-cubin` behind. flashinfer tolerates the cubin package being
*absent* (it then fetches cubins at runtime), but a *mismatched* version is a
hard error at engine startup, in `flashinfer/jit/env.py`:

```
RuntimeError: flashinfer-cubin version (X) does not match flashinfer version (Y).
```

(First hit: v0.23.0 -> v0.27.1 bump, 2026-08-25: python 0.6.16.post3 vs
cubin 0.6.12.)

Fix — reinstall the pinned cubin from the flashinfer index:

```bash
uv pip install "flashinfer-cubin==$(grep -oP 'flashinfer-cubin==\K[0-9a-z.+]+' requirements/cuda.txt)" \
    --extra-index-url https://flashinfer.ai/whl/
```

Sanity check that the venv matches every pin in `requirements/cuda.txt`
(should print nothing but the last line):

```bash
.venv/bin/python - <<'EOF'
import re
from pathlib import Path
sp = sorted(Path(".venv/lib").glob("python3.*"))[0] / "site-packages"
pins = {}
for line in Path("requirements/cuda.txt").read_text().splitlines():
    line = line.split("#")[0].strip()
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=)\s*([\w.+\-]+)", line)
    if m:
        pins[m.group(1).lower().replace("_", "-")] = (m.group(2), m.group(3))
have = {}
for d in sp.glob("*.dist-info"):
    n, _, v = d.name[:-10].replace("_", "-").lower().rpartition("-")
    have[n] = v
for n, (op, v) in sorted(pins.items()):
    if n not in have or (op == "==" and have[n] != v):
        print(f"STALE: {n} installed={have.get(n)} want {op}{v}")
print("pin sweep done")
EOF
```
