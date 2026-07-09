import json, requests
from bs4 import BeautifulSoup

out = open("./data/raw/debug_out.txt", "w", encoding="utf-8")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    out.write(line + "\n")


short = [
    json.loads(l)
    for l in open("./data/raw/nuforc_enrich_checkpoint_TEST.jsonl")
    if len((json.loads(l).get("Full_Text") or "")) < 120
]
log(f"{len(short)} short rows total")

sid = short[0]["id"]
url = f"https://nuforc.org/sighting/?id={sid}"
html = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    },
).text

soup = BeautifulSoup(html, "html.parser")
primary = soup.find(id="primary")
log("id:", sid)
log("has #primary container? ->", primary is not None)
log("raw html length ->", len(html))
log("text in #primary ->", len(primary.get_text()) if primary else 0)
log("text in whole page ->", len(soup.get_text()))

# dump the region around the narrative so I can see the structure
i = html.find("Characteristics")
log("\n---- HTML around Characteristics ----")
log(html[i : i + 2500] if i != -1 else "Characteristics not found in raw html")

out.close()
print("\nwrote ./data/raw/debug_out.txt")
