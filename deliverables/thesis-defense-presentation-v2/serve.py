#!/usr/bin/env python3
"""Serve the defense deck on the local network and drive it from a phone.

    python3 serve.py              # port 8000
    python3 serve.py --port 9000

Open the printed laptop address in Chrome/Edge (the audience window, full screen on the
projector) and scan the QR code with the phone: it opens the phone presenter view (current
and next slide, notes, timer, next/previous buttons) which drives the audience window.

Only the Python standard library is used, so the folder can be copied to any laptop.
Static files are open to the network; the control API needs the token in the QR link
(requests from the laptop itself are always allowed).
"""
import argparse
import email.utils
import json
import mimetypes
import os
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parent
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")


# --------------------------------------------------------------------------------------------
# QR code (byte mode, error correction level M, versions 1-10), enough for a LAN address
# --------------------------------------------------------------------------------------------
# version: (EC codewords per block, [(blocks, data codewords per block), ...])
_QR_M = {
    1: (10, [(1, 16)]), 2: (16, [(1, 28)]), 3: (26, [(1, 44)]), 4: (18, [(2, 32)]),
    5: (24, [(2, 43)]), 6: (16, [(4, 27)]), 7: (18, [(4, 31)]), 8: (22, [(2, 38), (2, 39)]),
    9: (22, [(3, 36), (2, 37)]), 10: (26, [(4, 43), (1, 44)]),
}
_ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
          7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50]}

_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _rs_ec(data, n):
    gen = [1]
    for i in range(n):
        nxt = [0] * (len(gen) + 1)
        for j, g in enumerate(gen):
            nxt[j] ^= g
            if g:
                nxt[j + 1] ^= _EXP[_LOG[g] + i]
        gen = nxt
    rem = list(data) + [0] * n
    for i in range(len(data)):
        c = rem[i]
        if c:
            for j in range(1, len(gen)):
                if gen[j]:
                    rem[i + j] ^= _EXP[_LOG[gen[j]] + _LOG[c]]
    return rem[len(data):]


def _bch(value, poly, bits):
    v = value << bits
    plen = poly.bit_length()
    while v.bit_length() >= plen:
        v ^= poly << (v.bit_length() - plen)
    return (value << bits) | v


def qr_matrix(text, mask=None):
    """Return a list of rows of booleans (True = dark), without the quiet zone."""
    data = text.encode("utf8")
    for ver in range(1, 11):
        ecn, groups = _QR_M[ver]
        cap = sum(b * d for b, d in groups)
        cbits = 8 if ver < 10 else 16
        if 4 + cbits + 8 * len(data) <= cap * 8:
            break
    else:
        raise ValueError("text too long for the built-in QR encoder")
    bits = [0, 1, 0, 0] + [(len(data) >> i) & 1 for i in range(cbits - 1, -1, -1)]
    for byte in data:
        bits += [(byte >> i) & 1 for i in range(7, -1, -1)]
    bits += [0] * min(4, cap * 8 - len(bits))
    bits += [0] * (-len(bits) % 8)
    cw = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = 0
    while len(cw) < cap:
        cw.append((0xEC, 0x11)[pad])
        pad ^= 1
    blocks, k = [], 0
    for nb, nd in groups:
        for _ in range(nb):
            blocks.append(cw[k:k + nd])
            k += nd
    ecs = [_rs_ec(b, ecn) for b in blocks]
    final = []
    for i in range(max(len(b) for b in blocks)):
        final += [b[i] for b in blocks if i < len(b)]
    for i in range(ecn):
        final += [e[i] for e in ecs]

    n = 17 + 4 * ver
    mod = [[False] * n for _ in range(n)]
    fixed = [[False] * n for _ in range(n)]

    def put(r, c, v):
        mod[r][c] = v
        fixed[r][c] = True

    for r0, c0 in ((0, 0), (0, n - 7), (n - 7, 0)):
        for r in range(-1, 8):
            for c in range(-1, 8):
                if 0 <= r0 + r < n and 0 <= c0 + c < n:
                    d = max(abs(r - 3), abs(c - 3))
                    put(r0 + r, c0 + c, d != 2 and d != 4)
    for i in range(8, n - 8):
        put(6, i, i % 2 == 0)
        put(i, 6, i % 2 == 0)
    pos = _ALIGN[ver]
    for r in pos:
        for c in pos:
            if (r == 6 and c in (6, n - 7)) or (r == n - 7 and c == 6):  # finder corners
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    put(r + dr, c + dc, max(abs(dr), abs(dc)) != 1)
    put(n - 8, 8, True)  # dark module
    for i in range(9):  # reserve format areas
        if not fixed[8][i]:
            fixed[8][i] = True
        if not fixed[i][8]:
            fixed[i][8] = True
    for i in range(8):
        fixed[8][n - 1 - i] = True
        fixed[n - 1 - i][8] = True
    if ver >= 7:
        vinfo = _bch(ver, 0x1F25, 12)
        for i in range(18):
            b = bool((vinfo >> i) & 1)
            put(n - 11 + i % 3, i // 3, b)
            put(i // 3, n - 11 + i % 3, b)

    stream = []
    for byte in final:
        stream += [bool((byte >> i) & 1) for i in range(7, -1, -1)]
    k, upward, col = 0, True, n - 1
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(n - 1, -1, -1) if upward else range(n)
        for r in rows:
            for c in (col, col - 1):
                if not fixed[r][c]:
                    mod[r][c] = stream[k] if k < len(stream) else False
                    k += 1
        upward = not upward
        col -= 2

    masks = [
        lambda r, c: (r + c) % 2 == 0, lambda r, c: r % 2 == 0, lambda r, c: c % 3 == 0,
        lambda r, c: (r + c) % 3 == 0, lambda r, c: (r // 2 + c // 3) % 2 == 0,
        lambda r, c: (r * c) % 2 + (r * c) % 3 == 0, lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
        lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0,
    ]

    def build(m):
        g = [row[:] for row in mod]
        f = masks[m]
        for r in range(n):
            for c in range(n):
                if not fixed[r][c] and f(r, c):
                    g[r][c] = not g[r][c]
        fmt = _bch((0b00 << 3) | m, 0x537, 10) ^ 0x5412
        fb = [bool((fmt >> i) & 1) for i in range(15)]  # fb[0] is the least significant bit
        for i in range(15):
            # first copy, around the top-left finder
            if i < 6:
                g[i][8] = fb[i]
            elif i < 8:
                g[i + 1][8] = fb[i]
            else:
                g[8][7 if i == 8 else 14 - i] = fb[i]
            # second copy, split between the other two finders
            if i < 8:
                g[8][n - 1 - i] = fb[i]
            else:
                g[n - 15 + i][8] = fb[i]
        g[n - 8][8] = True
        return g

    def penalty(g):
        p = 0
        for lines in (g, list(zip(*g))):
            for row in lines:
                run, prev = 0, None
                for v in row:
                    if v == prev:
                        run += 1
                    else:
                        if run >= 5:
                            p += run - 2
                        run, prev = 1, v
                if run >= 5:
                    p += run - 2
                s = "".join("1" if v else "0" for v in row)
                for pat in ("10111010000", "00001011101"):
                    i = s.find(pat)
                    while i != -1:
                        p += 40
                        i = s.find(pat, i + 1)
        for r in range(n - 1):
            for c in range(n - 1):
                if g[r][c] == g[r][c + 1] == g[r + 1][c] == g[r + 1][c + 1]:
                    p += 3
        dark = sum(map(sum, g))
        p += 10 * (abs(dark * 20 - n * n * 10) // (n * n))
        return p

    if mask is not None:
        return build(mask)
    return min((build(m) for m in range(8)), key=penalty)


def qr_terminal(text):
    m = qr_matrix(text)
    n = len(m)
    q = 2
    grid = [[False] * (n + 2 * q) for _ in range(q)] + \
           [[False] * q + row + [False] * q for row in m] + \
           [[False] * (n + 2 * q) for _ in range(q + 1)]
    out = []
    # light-on-dark terminals and dark-on-light ones both scan when we paint the light modules
    for r in range(0, len(grid) - 1, 2):
        line = ""
        for c in range(n + 2 * q):
            top, bot = not grid[r][c], not grid[r + 1][c]
            line += "█" if top and bot else "▀" if top else "▄" if bot else " "
        out.append(line)
    return "\n".join(out)


def qr_svg(text, px=8):
    m = qr_matrix(text)
    n = len(m) + 8
    rects = "".join(f'<rect x="{c + 4}" y="{r + 4}" width="1" height="1"/>'
                    for r, row in enumerate(m) for c, v in enumerate(row) if v)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="{n * px}" height="{n * px}" '
            f'shape-rendering="crispEdges"><rect width="{n}" height="{n}" fill="#fff"/><g fill="#000">{rects}</g></svg>')


# --------------------------------------------------------------------------------------------
# relay between the audience window(s) and the phone(s): Server-Sent Events down, POST up
# --------------------------------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = {}  # queue -> role
        self.state = None

    def add(self, role):
        q = queue.Queue()
        with self.lock:
            self.clients[q] = role
        self.status()
        return q

    def remove(self, q):
        with self.lock:
            self.clients.pop(q, None)
        self.status()

    def send(self, event, data, role=None):
        msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf8")
        with self.lock:
            targets = [q for q, r in self.clients.items() if role is None or r == role]
        for q in targets:
            q.put(msg)

    def status(self):
        with self.lock:
            roles = list(self.clients.values())
        self.send("status", {"mains": roles.count("main"), "remotes": roles.count("remote")})


HUB = Hub()
TOKEN = ""


class Handler(BaseHTTPRequestHandler):
    server_version = "DeckServer/1"

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/") and "/api/events" not in self.path:
            return
        if getattr(self, "_status", 200) >= 400:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def send_response(self, code, message=None):
        self._status = code
        super().send_response(code, message)

    def _local(self):
        return self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    def _authorised(self, qs):
        return self._local() or (qs.get("k", [""])[0] == TOKEN)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- API ----
    def do_POST(self):
        url = urlsplit(self.path)
        qs = parse_qs(url.query)
        if not url.path.startswith("/api/"):
            return self._json(404, {"error": "not found"})
        if not self._authorised(qs):
            return self._json(403, {"error": "bad token"})
        if url.path == "/api/notes" and not self._local():
            return self._json(403, {"error": "notes can only be edited on the laptop"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            msg = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad json"})
        if url.path == "/api/state":      # audience window -> phones
            HUB.state = msg
            HUB.send("state", msg, "remote")
        elif url.path == "/api/video":    # audience window -> phones, video position only
            if HUB.state is not None:
                HUB.state["video"] = msg.get("video")
            HUB.send("video", msg, "remote")
        elif url.path == "/api/cmd":      # phone -> audience window(s)
            HUB.send("cmd", msg, "main")
        elif url.path == "/api/notes":    # laptop presenter window -> content/fa.json
            err = save_note(msg)
            if err:
                return self._json(400, {"error": err})
            HUB.send("notes", {"key": msg["key"], "kind": msg["kind"], "text": msg["text"]}, "remote")
        else:
            return self._json(404, {"error": "not found"})
        self._json(200, {"ok": True})

    def _events(self, qs):
        role = "main" if qs.get("role", [""])[0] == "main" else "remote"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = HUB.add(role)
        try:
            self.wfile.write(b"retry: 1500\n\n")
            if role == "remote" and HUB.state is not None:
                self.wfile.write(f"event: state\ndata: {json.dumps(HUB.state)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=4)  # a ping every few seconds notices a closed window quickly
                except queue.Empty:
                    msg = b": ping\n\n"
                self.wfile.write(msg)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            HUB.remove(q)

    # ---- static files (with Range, so video seeking works) ----
    def do_GET(self):
        self._get(True)

    def do_HEAD(self):
        self._get(False)

    def _get(self, body):
        url = urlsplit(self.path)
        qs = parse_qs(url.query)
        if url.path == "/api/ping":
            return self._json(200 if self._authorised(qs) else 403, {"deck": True})
        if url.path == "/api/events":
            if not self._authorised(qs):
                return self._json(403, {"error": "bad token"})
            return self._events(qs)
        if url.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if url.path == "/qr":
            return self._qr_page()
        rel = unquote(url.path).lstrip("/") or "index.html"
        path = (ROOT / rel).resolve()
        if ROOT not in path.parents and path != ROOT:
            return self._json(403, {"error": "forbidden"})
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file() or path.name.endswith(".py"):
            return self._json(404, {"error": "not found"})
        st = path.stat()
        size = st.st_size
        lm = email.utils.formatdate(st.st_mtime, usegmt=True)
        ims = self.headers.get("If-Modified-Since")
        if ims and ims == lm and not self.headers.get("Range"):
            self.send_response(304)
            self.end_headers()
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "text/javascript"):
            ctype += "; charset=utf-8"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            try:
                if a:
                    start, end = int(a), (int(b) if b else size - 1)
                else:
                    start, end = max(0, size - int(b)), size - 1
            except ValueError:
                rng = None
            end = min(end, size - 1)
            if rng and (start > end or start >= size):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", lm)
        self.send_header("Cache-Control", "no-cache")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        if not body:
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _qr_page(self):
        port = self.server.server_address[1]
        rows = "".join(
            f'<figure><div class="q">{qr_svg(remote_url(ip, port), 7)}</div><figcaption dir="ltr">{remote_url(ip, port)}</figcaption></figure>'
            for ip in lan_ips())
        html = f"""<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Phone remote</title>
<style>@font-face{{font-family:Vazirmatn;src:url(assets/fonts/Vazirmatn-Bold.woff2)}}
body{{margin:0;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;
background:#0c1220;color:#e8edf7;font-family:Vazirmatn,sans-serif}}h1{{margin:0;font-size:26px}}
.row{{display:flex;flex-wrap:wrap;gap:28px;justify-content:center}}figure{{margin:0;text-align:center}}
.q svg{{display:block;border-radius:10px}}figcaption{{margin-top:10px;font:15px monospace;color:#a6d8f5}}
p{{color:#9fb0cf;margin:0;font-size:16px}}a{{color:#ffd479}}</style></head><body>
<h1>کنترل ارائه با گوشی</h1><div class="row">{rows}</div>
<p>گوشی و این رایانه باید در یک شبکه باشند. <a href="/">پنجره اصلی ارائه</a></p></body></html>"""
        body = html.encode("utf8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


NOTES_LOCK = threading.Lock()


def save_note(msg):
    """Write one slide's notes (or summary) into content/fa.json and rebuild index.html. Returns an error text or None."""
    key, kind, text = msg.get("key"), msg.get("kind"), msg.get("text")
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", key) or kind not in ("notes", "sum") or not isinstance(text, str):
        return "bad request"
    path = ROOT / "content" / "fa.json"
    with NOTES_LOCK:
        old = path.read_text(encoding="utf8")
        fa = json.loads(old)
        if f"{key}.notes" not in fa:
            return "unknown slide"
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if kind == "sum" and not text:
            fa.pop(f"{key}.sum", None)
        else:
            fa[f"{key}.{kind}"] = text
        new = json.dumps(fa, ensure_ascii=False, indent=1)   # the file's own layout: indent 1, no final newline
        if new == old:
            return None
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(new, encoding="utf8")
        os.replace(tmp, path)
        r = subprocess.run([sys.executable, str(ROOT / "build.py")], cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            path.write_text(old, encoding="utf8")
            subprocess.run([sys.executable, str(ROOT / "build.py")], cwd=str(ROOT), capture_output=True, text=True)
            return "build failed: " + (r.stderr or r.stdout).strip()[-200:]
        print(f"  notes saved: {key}.{kind}")
    return None


def lan_ips():
    ips = []
    try:  # the address the default route uses (no packet is sent)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127.") and not ip.startswith("172.17."):
                ips.append(ip)
    except OSError:
        pass
    return ips or ["127.0.0.1"]


def remote_url(ip, port):
    return f"http://{ip}:{port}/?remote=1&k={TOKEN}"


def main():
    global TOKEN
    try:  # the QR uses block characters; Windows consoles default to a code page without them
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except (AttributeError, ValueError):
        pass
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--token", default=None, help="fixed access token for the phone link (default: random)")
    a = ap.parse_args()
    TOKEN = a.token or secrets.token_hex(8)  # 64 bits: cannot be guessed by trying links

    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        srv = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as e:
        sys.exit(f"cannot listen on port {a.port}: {e}. Try --port 8080")
    port = srv.server_address[1]
    ips = lan_ips()
    print()
    print(f"  Audience window (this laptop, full screen on the projector):  http://localhost:{port}/")
    print(f"  QR code page for the phone (open on this laptop):           http://localhost:{port}/qr")
    for ip in ips:
        print(f"  Phone presenter link:  {remote_url(ip, port)}")
    print()
    try:
        print(qr_terminal(remote_url(ips[0], port)))
    except Exception as e:  # never let the QR stop the server
        print(f"  (QR code unavailable: {e})")
    print("\n  Phone and laptop must be on the same Wi-Fi (or the laptop on the phone's hotspot).")
    print("  Ctrl+C stops the server.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
