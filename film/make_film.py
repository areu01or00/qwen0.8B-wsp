"""Render film.html to an mp4: headless Chromium draws each frame from t = i/fps, posts the raw RGBA pixels here,
and they go straight into ffmpeg (the CUDA-Vid-Studio world/film.py pattern). Reports progress in OUT.progress.
    python film/make_film.py OUT.mp4 [--fps 30] [--t0 0 --t1 END]"""
import json, os, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]; out = os.path.abspath(a[0]); opt = {a[i]: a[i + 1] for i in range(1, len(a) - 1) if a[i].startswith("--")}
fps = float(opt.get("--fps", 30)); end = json.load(open(os.path.join(HERE, "film.json")))["script"]["end"]
t0, t1 = float(opt.get("--t0", 0)), float(opt.get("--t1", end))
job = json.dumps({"fps": fps, "from": int(t0 * fps), "to": int(t1 * fps)}).encode(); total = int((t1 - t0) * fps)
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "1920x1080", "-r", str(fps), "-i", "-",
                       "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
done, n, tic = threading.Event(), [0], time.time()

class Hd(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def send(self, body, ctype):
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/job.json": return self.send(job, "application/json")
        f = os.path.join(HERE, (p.lstrip("/") or "film.html"))
        self.send(open(f, "rb").read(), "text/html; charset=utf-8" if f.endswith(".html") else "application/json")
    def do_POST(self):
        b = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/frame":
            ff.stdin.write(b); n[0] += 1
            if n[0] % 30 == 0:
                el = time.time() - tic
                open(out + ".progress", "w").write(f"{n[0]}/{total} frames, {el:.0f}s, eta {el / n[0] * (total - n[0]):.0f}s\n")
        elif self.path == "/done": done.set()
        self.send(b"ok", "text/plain")

srv = ThreadingHTTPServer(("127.0.0.1", 0), Hd); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
ch = subprocess.Popen(["chromium", "--headless=new", "--no-sandbox", "--use-angle=gl", "--window-size=1920,1080", "--hide-scrollbars",
                       f"--user-data-dir=/tmp/wsp-film-{port}", f"http://127.0.0.1:{port}/film.html"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
done.wait(); ff.stdin.close(); ff.wait(); ch.terminate(); srv.shutdown()
open(out + ".progress", "w").write(f"DONE {n[0]}/{total} frames in {time.time() - tic:.0f}s\n"); print("wrote", out, n[0], "frames")
