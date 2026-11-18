"""Check a submission zip against the ranked UrbanTwin entry, or repack predictions and frames into one.

    # 1. check the published submission zip (md5 of each part)
    python tools/verify_submission.py check --track TRACK --zip submission.zip

    # 2. repack predictions.json + 50 synthetic frames the way utl.package does, then check the result
    python tools/verify_submission.py repack --track TRACK --pred predictions.json --synthetic DIR \
        --out repacked.zip --name "Sadbodh Sharma" --date 2026-11-15

The whole-zip md5 cannot match, because zip entries carry their write time; each entry is compared instead.
"""
import argparse
import datetime
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utl.package import validate  # noqa: E402
from utl.tracks import DECLARATION  # noqa: E402

# The ranked entries (Final phase, closing leaderboard): v2x_v35d (submission 1054261), lumpi_v62d (submission 1054262).
EXPECTED = {
    "v2x": dict(
        predictions="be6122b96cf619280f260c1e34244d24",
        synthetic="8a31aefaf1bfe96de6e4e6a04c471116",
        declaration="b694dd4eed70290e2828cb767d8473d0",
    ),
    "lumpi": dict(
        predictions="500b2d308d4df3d2c3e47abb6f25ab30",
        synthetic="811425d61c2c793c61101278299a032e",
        declaration="b694dd4eed70290e2828cb767d8473d0",
    ),
}
POINTER = ('The uses of real data in this submission are listed in the "Real-data use" section of the '
           "accompanying report.")


def md5(b):
    return hashlib.md5(b).hexdigest()


def check(track, zpath):
    z = zipfile.ZipFile(zpath)
    syn = hashlib.md5()
    for i in range(1, 51):
        syn.update(z.read(f"synthetic/{i:04d}.bin"))
    got = dict(predictions=md5(z.read("predictions.json")), synthetic=syn.hexdigest(),
               declaration=md5(z.read("declaration.txt")))
    ok = True
    for k, want in EXPECTED[track].items():
        same = got[k] == want
        ok &= same
        print(f"{k:12s} {got[k]}  expected {want}  {'OK' if same else 'DIFFERENT'}")
    print("ALL MATCH" if ok else "MISMATCH")
    return ok


def repack(a):
    preds = json.loads(Path(a.pred).read_text())
    preds = {f: sorted(d, key=lambda x: -x["score"]) for f, d in preds.items()}   # as utl.package: stable sort, highest first
    syn = sorted(Path(a.synthetic).glob("*.bin"))
    validate(a.track, preds, syn)
    date = a.date or datetime.date.today().isoformat()
    decl = f"{DECLARATION}\n{POINTER}\n\nName: {a.name}\nDate: {date}\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i, p in enumerate(syn, 1):
            z.write(p, f"synthetic/{i:04d}.bin")
        z.writestr("predictions.json", json.dumps(preds))
        z.writestr("declaration.txt", decl)
    print(f"wrote {out}")
    return check(a.track, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--track", required=True, choices=list(EXPECTED))
    c.add_argument("--zip", required=True)
    r = sub.add_parser("repack")
    r.add_argument("--track", required=True, choices=list(EXPECTED))
    r.add_argument("--pred", required=True)
    r.add_argument("--synthetic", required=True, help="directory with 0001.bin .. 0050.bin")
    r.add_argument("--out", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--date", help="declaration date, YYYY-MM-DD (the ranked entries used 2026-11-15)")
    a = ap.parse_args()
    ok = check(a.track, a.zip) if a.cmd == "check" else repack(a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
