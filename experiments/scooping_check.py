"""Monthly scooping check (brief §10.1): search arXiv for DP/BNP-splatting papers.

Appends a dated block to stdout (for the project's literature-watch log).
Uses only stdlib. Run: python experiments/scooping_check.py
"""

import re
import time
import urllib.parse
import urllib.request
from datetime import date

QUERIES = [
    "dirichlet process splatting",
    "nonparametric gaussian splatting",
    "bayesian gaussian splatting",
    "infinite mixture splatting",
    "dirichlet process gaussian splatting",
]

# Known non-scooping hits to suppress from the alert list (still counted).
KNOWN = {"2604.02696", "2603.08499", "2410.03592", "2504.01844"}

API = "https://export.arxiv.org/api/query?search_query=all:%22{q}%22&max_results=20&sortBy=submittedDate&sortOrder=descending"


def search(query):
    url = API.format(q=urllib.parse.quote_plus(query).replace("+", "+"))
    with urllib.request.urlopen(url, timeout=30) as r:
        xml = r.read().decode()
    hits = []
    for m in re.finditer(r"<entry>.*?</entry>", xml, re.S):
        e = m.group(0)
        aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e).group(1)
        title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip()
        pub = re.search(r"<published>([^T<]+)", e).group(1)
        hits.append((aid, pub, title))
    return hits


def main():
    print(f"### Scooping check {date.today().isoformat()} (brief §10.1)")
    new_alerts = 0
    for q in QUERIES:
        hits = search(q)
        fresh = [h for h in hits if h[0].split("v")[0] not in KNOWN]
        status = f"{len(hits)} hit(s), {len(fresh)} new" if hits else "0 hits"
        print(f'- "{q}": {status}')
        for aid, pub, title in fresh:
            print(f"    - **NEW** {aid} ({pub}) {title}")
            new_alerts += 1
        time.sleep(1)
    verdict = "no direct DP/BNP-splatting paper found" if new_alerts == 0 else f"{new_alerts} NEW hit(s) — review for scooping (brief §10.1 pivot plan)"
    print(f"- Verdict: {verdict}.")


if __name__ == "__main__":
    main()
