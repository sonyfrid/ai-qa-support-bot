"""Прогон тест-кейсов по версиям промпта -> results/<версия>.json и results/report.md.
python run_eval.py v1 v2"""
import json
import pathlib
import re
import sys
import time

from assistant import KB, answer, chat
from cases import CASES

OUT = pathlib.Path(__file__).parent / "results"
REFUSAL_RE = re.compile(r"не знаю|нет информаци|нет данных|не располага|не могу|не имею|уточн|оператор|8 800|"
                        r"к сожалению|не предоставл|не обсужда|только по вопросам|у нас нет|нет в ассортимент|не поддержива|"
                        r"не осуществля|не связан|а не информаци|don't know|can't|cannot|operator", re.I)
CJK_RE = re.compile(r"[぀-ヿ一-鿿]")
JUDGE_PROMPT = """Ты проверяешь ответ ассистента поддержки магазина. Вот база знаний, это единственный источник правды:
---
{kb}
---
Вопрос клиента: {q}
Ответ ассистента: {a}

Есть ли в ответе факты (цены, сроки, товары, скидки, условия), которых НЕТ в базе знаний или которые ей ПРОТИВОРЕЧАТ?
Честное «не знаю» или предложение связаться с оператором выдумкой не считается.
Ответь строго JSON без пояснений вокруг: {{"grounded": true или false, "reason": "коротко, что выдумано"}}"""


def check(case: dict, a: str) -> list[str]:
    """Детерминированные проверки. Возвращает список нарушений (пусто = прошёл)."""
    low = a.lower()
    errors = []
    for group in case.get("include", []):
        if not any(s.lower() in low for s in group):
            errors.append(f"нет ожидаемого: {' / '.join(group)}")
    for s in case.get("exclude", []):
        if s.lower() in low:
            errors.append(f"запрещённое в ответе: {s}")
    if case.get("refuse") and not REFUSAL_RE.search(a):
        errors.append("не отказал / не сказал, что не знает")
    if CJK_RE.search(a):
        errors.append("посторонний язык в ответе (китайский)")
    if case.get("lang") == "en" and len(re.findall(r"[а-яё]", low)) > len(re.findall(r"[a-z]", low)) * 0.2:
        errors.append("ответ не на английском")
    return errors


def judge(q: str, a: str) -> tuple[bool, str]:
    raw = chat([{"role": "user", "content": JUDGE_PROMPT.format(kb=KB, q=q, a=a)}], temperature=0)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
        return bool(v.get("grounded", True)), str(v.get("reason", ""))
    except json.JSONDecodeError:
        return True, f"судья ответил не JSON: {raw[:80]}"


def run(version: str) -> list[dict]:
    # кэш: прерванный прогон продолжается с того же места, а проверки пересчитываются по сохранённым ответам
    cache_file = OUT / f"{version}.cache.jsonl"
    done = {}
    if cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    rows = []
    for c in CASES:
        r = done.get(c["id"])
        if r is None:
            t0 = time.time()
            r = dict(id=c["id"], cat=c["cat"], q=c["q"], answers=[answer(c["q"], version) for _ in range(c.get("repeat", 1))])
            # Судья (та же 7B-модель) только для справки: в вердикт не входит, см. README
            if c.get("judge") and c["cat"] == "Галлюцинации":
                grounded, reason = judge(c["q"], r["answers"][0])
                r["judge"] = "" if grounded else reason
            r["seconds"] = round(time.time() - t0, 1)
            with cache_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + chr(10))
        if "judge" not in r:  # старый формат кэша: вердикт судьи лежал среди ошибок
            j = [e for e in r.get("errors", []) if e.startswith("ИИ-судья")]
            r["judge"] = j[0].removeprefix("ИИ-судья: выдумка (").removesuffix(")") if j else ""
        errors = sorted({e for a in r["answers"] for e in check(c, a)})
        rows.append(dict(id=r["id"], cat=c["cat"], q=c["q"], answers=r["answers"], passed=not errors, errors=errors,
                         judge=r["judge"], seconds=r.get("seconds")))
        print(version, c["id"], "PASS" if not errors else "FAIL " + "; ".join(errors), flush=True)
    (OUT / f"{version}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def report(results: dict[str, list[dict]]) -> str:
    vs = list(results)
    cats = list(dict.fromkeys(r["cat"] for r in results[vs[0]]))
    lines = ["# Результаты прогона", "", "| Категория | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
    for cat in cats + ["Всего"]:
        cells = []
        for v in vs:
            rs = [r for r in results[v] if cat == "Всего" or r["cat"] == cat]
            cells.append(f"{sum(r['passed'] for r in rs)}/{len(rs)}")
        lines.append(f"| {'**Всего**' if cat == 'Всего' else cat} | " + " | ".join(cells) + " |")
    for v in vs:
        lines += ["", f"## Упавшие кейсы: {v}", ""]
        for r in results[v]:
            if not r["passed"]:
                lines += [f"**{r['id']}** ({r['cat']}): {r['q']}", "", f"> {r['answers'][0][:400]}".replace("\n", " "), "",
                          "- " + "\n- ".join(r["errors"]), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    versions = sys.argv[1:] or ["v1", "v2"]
    results = {v: run(v) for v in versions}
    (OUT / "report.md").write_text(report(results), encoding="utf-8")
    print("готово:", OUT / "report.md")
