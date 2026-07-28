import os
import math
import sqlite3
import collections
import re
from datetime import datetime

from ecopulse.config import cfg

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_ROOT, "ecopulse", cfg.DB_PATH)
_REF_CSV = os.path.join(_ROOT, "ecopulse", cfg.REFERENCE_DATA_PATH)
_RPT_DIR = os.path.join(_ROOT, "ecopulse", "monitoring")
os.makedirs(_RPT_DIR, exist_ok=True)


def _tokenize(text):
    return re.findall(r"[а-яёa-z]{3,}", text.lower())

def _mean(v): return sum(v) / len(v) if v else 0.0
def _std(v):
    if len(v) < 2: return 0.0
    m = _mean(v)
    return math.sqrt(sum((x-m)**2 for x in v) / len(v))

def _top_words(texts, n=20):
    c = collections.Counter()
    for t in texts: c.update(_tokenize(t))
    return dict(c.most_common(n))

def _psi(ref, cur, bins=10):
    if not ref or not cur: return 0.0
    lo = min(min(ref), min(cur))
    hi = max(max(ref), max(cur)) + 1e-9
    step = (hi - lo) / bins
    def bucket(vals):
        counts = [0] * bins
        for v in vals:
            counts[min(int((v-lo)/step), bins-1)] += 1
        total = len(vals)
        return [max(c/total, 1e-6) for c in counts]
    r, c = bucket(ref), bucket(cur)
    return sum((c[i]-r[i]) * math.log(c[i]/r[i]) for i in range(bins))


def _load_db(days=7):
    if not os.path.exists(_DB_PATH): return []
    try:
        con  = sqlite3.connect(_DB_PATH)
        rows = con.execute(f"""
            SELECT text, label, model_score FROM posts
            WHERE parsed_at >= date('now', '-{days} days')
              AND text IS NOT NULL AND text != ''
        """).fetchall()
        con.close()
        return rows
    except Exception: return []


def _load_ref():
    if not os.path.exists(_REF_CSV): return []
    rows = []
    try:
        with open(_REF_CSV, encoding="utf-8") as f:
            hdr = f.readline().strip().split(",")
            ti  = hdr.index("text")  if "text"  in hdr else 0
            li  = hdr.index("label") if "label" in hdr else 1
            for line in f:
                p = line.strip().split(",", maxsplit=len(hdr)-1)
                if len(p) > max(ti, li):
                    rows.append((p[ti], p[li], 0.5))
    except Exception: pass
    return rows


def _analyze(rows):
    texts   = [r[0] for r in rows if r[0]]
    labels  = [int(r[1]) for r in rows if str(r[1]).lstrip('-').isdigit()]
    scores  = [float(r[2]) for r in rows if r[2]]
    lengths = [len(t.split()) for t in texts]
    return {"n": len(rows), "critical_rate": _mean(labels),
            "len_mean": _mean(lengths), "score_mean": _mean(scores),
            "top_words": _top_words(texts), "lengths": lengths, "scores": scores}


def _html(ref, cur, psi_len, psi_score, word_diff, drift):
    color = "#c0392b" if drift else "#27ae60"
    status = "⚠️ ДРЕЙФ ОБНАРУЖЕН" if drift else "✅ Данные стабильны"
    wrows = ""
    for w, d in sorted(word_diff.items(), key=lambda x: -abs(x[1]))[:15]:
        c = "#c0392b" if d > 0 else "#2980b9"
        wrows += f"<tr><td>{w}</td><td style='color:{c}'>{'+' if d>0 else ''}{d:.1%}</td></tr>"
    return f"""<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<title>EcoPulse Drift</title>
<style>body{{font-family:Arial;max-width:900px;margin:40px auto}}
.st{{color:{color};border:2px solid {color};padding:16px;border-radius:8px;font-size:20px;font-weight:bold}}
table{{border-collapse:collapse;width:100%}}th{{background:#1a5276;color:#fff;padding:10px}}
td{{padding:8px;border-bottom:1px solid #eee}}tr:nth-child(even){{background:#f4f8fb}}</style></head>
<body><h1>🌿 EcoPulse — Дрейф данных</h1>
<p>{datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
<div class="st">{status}</div>
<h2>Сравнение</h2><table>
<tr><th>Метрика</th><th>Эталон</th><th>Текущие</th><th>Изменение</th></tr>
<tr><td>Постов</td><td>{ref['n']}</td><td>{cur['n']}</td><td>—</td></tr>
<tr><td>Доля critical</td><td>{ref['critical_rate']:.1%}</td><td>{cur['critical_rate']:.1%}</td>
    <td>{cur['critical_rate']-ref['critical_rate']:+.1%}</td></tr>
<tr><td>Средняя длина</td><td>{ref['len_mean']:.1f}</td><td>{cur['len_mean']:.1f}</td>
    <td>{cur['len_mean']-ref['len_mean']:+.1f}</td></tr>
<tr><td>PSI длин</td><td colspan="2">{psi_len:.3f}</td><td>{'🔴 дрейф' if psi_len>0.2 else '✅'}</td></tr>
<tr><td>PSI скора</td><td colspan="2">{psi_score:.3f}</td><td>{'🔴 дрейф' if psi_score>0.2 else '✅'}</td></tr>
</table><h2>Лексика</h2><table><tr><th>Слово</th><th>Изменение</th></tr>{wrows}</table>
</body></html>"""


def check_drift(days=7):
    ref_rows = _load_ref()
    cur_rows = _load_db(days)

    if not ref_rows:
        print(f"[drift] эталон не найден: {_REF_CSV}")
        print("        Создай: cp ecopulse/data/train.csv ecopulse/data/train_reference.csv")
        return {"drift_detected": False, "reason": "no_reference"}

    if len(cur_rows) < 30:
        print(f"[drift] мало данных: {len(cur_rows)} (нужно >= 30)")
        return {"drift_detected": False, "reason": "not_enough_data"}

    ref = _analyze(ref_rows)
    cur = _analyze(cur_rows)

    psi_len   = _psi(ref["lengths"], cur["lengths"])
    psi_score = _psi(ref["scores"],  cur["scores"])

    ref_total = sum(ref["top_words"].values()) or 1
    cur_total = sum(cur["top_words"].values()) or 1
    all_words = set(ref["top_words"]) | set(cur["top_words"])
    word_diff = {w: cur["top_words"].get(w,0)/cur_total
                    - ref["top_words"].get(w,0)/ref_total for w in all_words}

    drift = (psi_len > 0.2 or psi_score > 0.2
             or abs(cur["critical_rate"] - ref["critical_rate"]) > 0.15)

    print(f"[drift] PSI длин={psi_len:.3f} скора={psi_score:.3f} drift={drift}")

    rpt = os.path.join(_RPT_DIR, "drift_report.html")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(_html(ref, cur, psi_len, psi_score, word_diff, drift))
    print(f"[drift] отчёт: {rpt}")

    return {"drift_detected": drift, "psi_len": psi_len, "psi_score": psi_score}


if __name__ == "__main__":
    print(check_drift())
