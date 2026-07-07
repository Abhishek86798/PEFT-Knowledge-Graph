"""Render graph.json as a single self-contained interactive HTML page.

The output (graph.html) needs no server and no internet: vis-network is inlined.
Design tells a story instead of dumping 86 nodes at once:

  * default view = the 13 methods (family-coloured) + 18 benchmarks;
  * each method carries a paper-count badge;
  * clicking a method reveals ONLY that method's APPLIES papers (progressive);
  * a side panel shows the clicked method's facts — introduced date, #papers,
    benchmarks (EVALUATED_ON), compared-against methods, and a link to its paper.

A reviewer opens the file and reads the graph as Method -> Papers -> Benchmarks
without touching JSON or code.

Usage:  python src/render_html.py   ->   writes graph.html at repo root
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "graph.json"
OUT = ROOT / "graph.html"
VENDOR = ROOT / "src" / "vendor" / "vis-network.min.js"

ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")

FAMILY_COLOR = {
    "additive-adapter": "#6c5ce7",
    "reparameterization": "#b0392a",
    "soft-prompt": "#159a7f",
    "selective": "#7f8c8d",
    "multiplicative": "#c98a2b",
}
BENCH_COLOR = "#3a3f45"
PAPER_COLOR = "#2b2f36"


def paper_url(pid: str) -> str:
    if not pid:
        return ""
    if ARXIV.match(pid):
        return f"https://arxiv.org/abs/{pid}"
    return f"https://www.semanticscholar.org/paper/{pid}"


def build_model(g: dict) -> dict:
    methods = {m["slug"]: m for m in g["methods"]}
    papers = {p["arxiv_id"]: p for p in g["papers"]}
    benches = {b["slug"]: b for b in g["benchmarks"]}
    intro = {e["to_method"]: e["from_paper"] for e in g["edges"]["INTRODUCES"]}

    applies = defaultdict(list)
    for e in g["edges"]["APPLIES"]:
        applies[e["to_method"]].append(e["from_paper"])
    evalon = defaultdict(list)
    for e in g["edges"]["EVALUATED_ON"]:
        evalon[e["from_method"]].append(benches[e["to_benchmark"]]["name"])
    compared = defaultdict(list)
    for e in g["edges"]["COMPARED_AGAINST"]:
        compared[e["from_method"]].append(methods[e["to_method"]]["name"])

    nodes, edges = [], []

    # Method nodes (skeleton) — the visual anchors: large boxes with a two-line
    # badge (name + paper count) so a reviewer spots LoRA/Adapters/etc. instantly.
    for slug, m in methods.items():
        n_papers = len(applies[slug])
        # Real newline (not "\\n"): vis.js multi-line labels need an actual \n.
        # Show the count only when > 0 so zero-application methods don't read as
        # dead ends -- they still carry EXTENDS/EVALUATED_ON/COMPARED_AGAINST edges.
        # Badge counts APPLIES papers specifically (papers that *run* the method),
        # not all papers touching it -- shown only when > 0 so 0-application methods
        # (which still have INTRODUCES/EXTENDS/EVALUATED_ON edges) look clean.
        badge = f"\n\U0001F4C4 applied in {n_papers}" if n_papers else ""
        nodes.append({
            "id": slug, "kind": "method",
            "label": f'{m["name"]}{badge}',
            "color": FAMILY_COLOR.get(m["family"], "#888"),
            "shape": "box",
            # Deliberately much larger than benchmarks/papers -> visual hierarchy.
            "size": 40 if m["is_family_root"] else 34,
        })

    # Benchmark nodes (always visible) — medium grey circles, clearly a tier below
    # methods.
    for slug, b in benches.items():
        nodes.append({
            "id": f"bench_{slug}", "kind": "benchmark",
            "label": b["name"], "color": BENCH_COLOR,
            "shape": "ellipse", "size": 16,
        })

    # Paper nodes (hidden until their method is clicked). One node per APPLIES paper.
    seen_paper = set()
    for slug, plist in applies.items():
        for pid in plist:
            nid = f"paper_{pid}"
            if nid not in seen_paper:
                seen_paper.add(nid)
                p = papers.get(pid, {})
                nodes.append({
                    "id": nid, "kind": "paper", "owner": slug,
                    "label": (p.get("title") or pid)[:34],
                    "color": PAPER_COLOR, "shape": "dot", "size": 7, "hidden": True,
                })

    # Edges (tagged; paper/applies edges hidden until reveal).
    for e in g["edges"]["EXTENDS"]:
        edges.append({"from": e["to_method"], "to": e["from_method"],
                      "etype": "extends", "color": "#8e9aa6"})
    for e in g["edges"]["EVALUATED_ON"]:
        edges.append({"from": e["from_method"], "to": f'bench_{e["to_benchmark"]}',
                      "etype": "eval", "color": "#6fbfa5"})
    for e in g["edges"]["COMPARED_AGAINST"]:
        edges.append({"from": e["from_method"], "to": e["to_method"],
                      "etype": "compared", "color": "#e08a6b", "dashes": True})
    for e in g["edges"]["APPLIES"]:
        edges.append({"from": f'paper_{e["from_paper"]}', "to": e["to_method"],
                      "etype": "applies", "color": "#7d74e0", "owner": e["to_method"]})

    # Per-method fact sheets for the side panel.
    facts = {}
    for slug, m in methods.items():
        pid = intro.get(slug)
        facts[slug] = {
            "name": m["name"], "family": m["family"],
            "introduced": m["introduced_date"][:4],
            "is_root": m["is_family_root"],
            "mechanism": m.get("mechanism", ""),
            "n_papers": len(applies[slug]),
            "benchmarks": sorted(set(evalon[slug])),
            "compared": sorted(set(compared[slug])),
            "paper_url": paper_url(pid),
            "paper_title": (papers.get(pid, {}).get("title") or "") if pid else "",
        }
    # Paper fact sheets (title + link).
    pfacts = {}
    for nid in seen_paper:
        pid = nid[len("paper_"):]
        p = papers.get(pid, {})
        pfacts[nid] = {"name": p.get("title") or pid, "url": paper_url(pid),
                       "venue": p.get("venue") or "", "year": p.get("year") or ""}

    stats = {"methods": len(methods), "papers": len(papers), "benchmarks": len(benches),
             "extends": len(g["edges"]["EXTENDS"]), "compared": len(g["edges"]["COMPARED_AGAINST"]),
             "eval": len(g["edges"]["EVALUATED_ON"]), "applies": len(g["edges"]["APPLIES"])}

    return {"nodes": nodes, "edges": edges, "facts": facts, "pfacts": pfacts, "stats": stats}


def render(g: dict) -> str:
    model = build_model(g)
    visjs = VENDOR.read_text(encoding="utf-8") if VENDOR.exists() else ""
    return _TEMPLATE.format(
        nodes=json.dumps(model["nodes"], ensure_ascii=False),
        edges=json.dumps(model["edges"], ensure_ascii=False),
        facts=json.dumps(model["facts"], ensure_ascii=False),
        pfacts=json.dumps(model["pfacts"], ensure_ascii=False),
        stats=json.dumps(model["stats"]),
        visjs=visjs,
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEFT Knowledge Graph</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:#14171c; color:#e6e9ee; overflow:hidden; }}
  header {{ padding:12px 20px; border-bottom:1px solid #262b32; display:flex;
           align-items:baseline; gap:16px; flex-wrap:wrap; }}
  header h1 {{ font-size:16px; margin:0; font-weight:650; }}
  header .sub {{ color:#8b93a0; font-size:12.5px; }}
  .stats {{ display:flex; gap:14px; flex-wrap:wrap; }}
  .stat {{ font-size:12px; color:#9aa3af; }} .stat b {{ color:#e6e9ee; font-size:14px; }}
  #search {{ margin-left:auto; }}
  #search input {{ background:#1c2128; border:1px solid #2f3841; color:#e6e9ee;
     padding:7px 12px; border-radius:7px; font-size:13px; width:190px; outline:none; }}
  #search input:focus {{ border-color:#5b6bd6; }}
  #toolbar {{ padding:8px 20px; display:flex; gap:14px; align-items:center;
             flex-wrap:wrap; border-bottom:1px solid #262b32; font-size:12.5px; }}
  #toolbar label {{ display:flex; gap:6px; align-items:center; cursor:pointer; }}
  .legend {{ display:flex; gap:11px; margin-left:auto; flex-wrap:wrap; font-size:11.5px; color:#9aa3af; }}
  .swatch {{ display:inline-block; width:11px; height:11px; border-radius:3px; vertical-align:middle; margin-right:4px; }}
  #wrap {{ display:flex; height:calc(100vh - 92px); }}
  #net {{ flex:1; height:100%; }}
  #panel {{ width:300px; border-left:1px solid #262b32; background:#171b21;
           padding:18px; overflow-y:auto; font-size:13px; line-height:1.5; }}
  #panel h2 {{ margin:0 0 2px; font-size:18px; }}
  #panel .fam {{ font-size:12px; color:#9aa3af; margin-bottom:14px; }}
  #panel .sec {{ margin:14px 0 4px; font-size:11px; letter-spacing:.06em;
                text-transform:uppercase; color:#7c8593; }}
  #panel .chip {{ display:inline-block; background:#232a32; border:1px solid #2f3841;
                 border-radius:5px; padding:2px 8px; margin:2px 3px 2px 0; font-size:12px; }}
  #panel .mech {{ color:#c3cad3; font-size:12.5px; font-style:italic; }}
  #panel a {{ display:inline-block; margin-top:6px; background:#2b3550; color:#aebfff;
             padding:7px 12px; border-radius:6px; text-decoration:none; font-weight:600; }}
  #panel .empty {{ color:#6b7280; }}
  #panel .placeholder {{ color:#7c8593; margin-top:40px; text-align:center; }}
  #panel .row {{ display:flex; align-items:baseline; gap:8px; margin:7px 0; }}
  #panel .row .ico {{ width:18px; flex:0 0 18px; }}
  #panel .row .rlab {{ color:#8b93a0; flex:1; }}
  #panel .row .rval {{ color:#e6e9ee; font-weight:600; text-align:right; }}
  #panel .ptitle {{ margin-top:8px; font-size:12px; color:#9aa3af; line-height:1.4; }}
  #panel .note {{ margin-top:14px; padding:10px 12px; background:#1e232b;
                 border:1px solid #2f3841; border-left:3px solid #c98a2b;
                 border-radius:6px; font-size:11.5px; color:#aab2bd; line-height:1.45; }}
  .hint {{ position:absolute; bottom:10px; left:20px; font-size:12px; color:#7c8593; }}
</style></head><body>
<header>
  <h1>PEFT Knowledge Graph</h1>
  <span class="sub">click a method to reveal its papers &middot; click any node for details</span>
  <div class="stats" id="stats"></div>
  <div id="search"><input id="searchbox" type="text" placeholder="Search methods…" autocomplete="off"></div>
</header>
<div id="toolbar">
  <strong style="color:#9aa3af;font-weight:600;">Edges:</strong>
  <label><input type="checkbox" id="cb-extends" checked> <span style="color:#8e9aa6">━</span> EXTENDS</label>
  <label><input type="checkbox" id="cb-eval" checked> <span style="color:#6fbfa5">━</span> EVALUATED_ON</label>
  <label><input type="checkbox" id="cb-compared" checked> <span style="color:#e08a6b">┄</span> COMPARED_AGAINST</label>
  <div class="legend">
    <span><span class="swatch" style="background:#6c5ce7"></span>Adapter</span>
    <span><span class="swatch" style="background:#b0392a"></span>LoRA</span>
    <span><span class="swatch" style="background:#159a7f"></span>Soft-prompt</span>
    <span><span class="swatch" style="background:#7f8c8d"></span>Selective</span>
    <span><span class="swatch" style="background:#c98a2b"></span>Multiplicative</span>
    <span><span class="swatch" style="background:#3a3f45"></span>Benchmark</span>
  </div>
</div>
<div id="wrap">
  <div id="net"></div>
  <div id="panel"><div class="placeholder">Click a method node to see its<br>papers, benchmarks, and lineage.</div></div>
</div>
<div class="hint" id="hint">Methods with a paper badge &gt; 0 expand on click. Uncheck edges to declutter.</div>

<script>{visjs}</script>
<script>
const RAW_NODES={nodes}, RAW_EDGES={edges}, FACTS={facts}, PFACTS={pfacts}, STATS={stats};
document.getElementById('stats').innerHTML =
 `<span class="stat"><b>${{STATS.methods}}</b> methods</span>
  <span class="stat"><b>${{STATS.papers}}</b> papers</span>
  <span class="stat"><b>${{STATS.benchmarks}}</b> benchmarks</span>
  <span class="stat"><b>${{STATS.extends+STATS.compared+STATS.eval+STATS.applies}}</b> edges</span>`;

function ensureVis(cb){{
  if(typeof vis!=='undefined') return cb();
  const s=document.createElement('script');
  s.src='https://unpkg.com/vis-network/standalone/umd/vis-network.min.js';
  s.onload=cb; document.head.appendChild(s);
}}

ensureVis(()=>{{
  const nodes=new vis.DataSet(RAW_NODES.map(n=>({{
    id:n.id, label:n.label, shape:n.shape, size:n.size, hidden:!!n.hidden,
    color:{{background:n.color, border:'#0f1216',
      highlight:{{background:n.color, border:'#ffffff'}}}},
    font:{{color:'#e6e9ee', size:n.kind==='method'?14:11, multi:true}},
    kind:n.kind, owner:n.owner||null
  }})));
  const allEdges=RAW_EDGES.map((e,i)=>({{
    id:i, from:e.from, to:e.to, arrows:'to', dashes:!!e.dashes,
    color:{{color:e.color, opacity:0.65}}, etype:e.etype, owner:e.owner||null,
    hidden:e.etype==='applies'
  }}));
  const edges=new vis.DataSet(allEdges);

  const net=new vis.Network(document.getElementById('net'), {{nodes,edges}}, {{
    physics:{{stabilization:true, barnesHut:{{gravitationalConstant:-9000,
      springLength:140, springConstant:0.04, avoidOverlap:0.1}}}},
    interaction:{{hover:true, tooltipDelay:150}}, nodes:{{borderWidth:1.5}}
  }});

  const revealed=new Set();
  function toggleMethodPapers(slug){{
    const show=!revealed.has(slug);
    if(show) revealed.add(slug); else revealed.delete(slug);
    nodes.forEach(n=>{{ if(n.kind==='paper' && n.owner===slug) nodes.update({{id:n.id,hidden:!show}}); }});
    allEdges.forEach(e=>{{ if(e.etype==='applies' && e.owner===slug) edges.update({{id:e.id,hidden:!show}}); }});
  }}

  function esc(s){{return (s||'').replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));}}
  function chips(arr){{ return arr.length? arr.map(x=>`<span class="chip">${{esc(x)}}</span>`).join('')
                        : '<span class="empty">none</span>'; }}
  function row(icon,label,val){{
    return `<div class="row"><span class="ico">${{icon}}</span>`+
           `<span class="rlab">${{label}}</span><span class="rval">${{val}}</span></div>`;
  }}
  function methodPanel(slug){{
    const f=FACTS[slug]; if(!f) return;
    const bench = f.benchmarks.length? chips(f.benchmarks) : '<span class="empty">none</span>';
    const comp  = f.compared.length?   chips(f.compared)   : '<span class="empty">none</span>';
    document.getElementById('panel').innerHTML =
     `<h2>${{esc(f.name)}}</h2>
      <div class="fam">${{f.is_root?'★ family root':'variant'}} · ${{esc(f.family)}}</div>
      <div class="mech">${{esc(f.mechanism)}}</div>
      ${{row('📅','Introduced', f.introduced)}}
      ${{row('📄','Applied in', f.n_papers? (f.n_papers+' papers <span style=\"color:#7c8593\">(click node to reveal)</span>')
             : '<span style=\"color:#7c8593\">0 in this corpus — see note</span>')}}
      ${{row('📚','Family', esc(f.family))}}
      <div class="sec">📊 Benchmarks (evaluated on)</div>${{bench}}
      <div class="sec">🔄 Compared against</div>${{comp}}
      ${{f.n_papers===0? `<div class="note">No <b>application</b> papers for this method
          in the corpus. The APPLIES pool was grown from citation links off
          LoRA-heavy seeds, so LoRA dominates (50) and several methods have 0 — a
          known coverage limitation (see approach.md). This method is still fully
          present via its EXTENDS / EVALUATED_ON / COMPARED_AGAINST edges above.</div>`:''}}
      ${{f.paper_url? `<div class="sec">Introducing paper</div>
          <a href="${{f.paper_url}}" target="_blank">Open introducing paper &rarr;</a>
          <div class="ptitle">${{esc(f.paper_title || f.name)}}</div>`:''}}`;
  }}
  function paperPanel(nid){{
    const f=PFACTS[nid]; if(!f) return;
    document.getElementById('panel').innerHTML =
     `<h2 style="font-size:15px">${{esc(f.name)}}</h2>
      <div class="fam">${{esc(String(f.venue))}} ${{f.year?('· '+f.year):''}}</div>
      <div class="sec">Application paper</div>
      <span style="color:#9aa3af">runs a tracked method unmodified (APPLIES)</span>
      ${{f.url? `<div class="sec">Source</div><a href="${{f.url}}" target="_blank">Open paper &rarr;</a>
          <div class="ptitle">${{esc(f.name)}}</div>`:''}}`;
  }}

  net.on('click', p=>{{
    if(!p.nodes.length) return;
    const n=nodes.get(p.nodes[0]);
    if(n.kind==='method'){{ methodPanel(n.id); toggleMethodPapers(n.id); }}
    else if(n.kind==='paper'){{ paperPanel(n.id); }}
    else if(n.kind==='benchmark'){{
      document.getElementById('panel').innerHTML =
        `<h2 style="font-size:16px">${{esc(n.label)}}</h2><div class="fam">benchmark</div>
         <div class="sec">Role</div><span style="color:#9aa3af">methods report EVALUATED_ON results here</span>`;
    }}
  }});

  const cbs={{extends:'cb-extends', eval:'cb-eval', compared:'cb-compared'}};
  Object.entries(cbs).forEach(([etype,id])=>{{
    document.getElementById(id).addEventListener('change', ev=>{{
      allEdges.forEach(e=>{{ if(e.etype===etype) edges.update({{id:e.id, hidden:!ev.target.checked}}); }});
    }});
  }});

  // Search: type a method name -> zoom, center, select, and open its panel.
  const methodIndex = Object.entries(FACTS).map(([slug,f])=>({{slug, name:f.name.toLowerCase()}}));
  function runSearch(q){{
    q=q.trim().toLowerCase(); if(!q) return;
    const hit = methodIndex.find(m=>m.name.startsWith(q)) || methodIndex.find(m=>m.name.includes(q));
    if(!hit) return;
    net.selectNodes([hit.slug]);
    net.focus(hit.slug, {{scale:1.3, animation:{{duration:600, easingFunction:'easeInOutQuad'}}}});
    methodPanel(hit.slug);
  }}
  const sb=document.getElementById('searchbox');
  sb.addEventListener('keydown', e=>{{ if(e.key==='Enter') runSearch(sb.value); }});
  sb.addEventListener('input', e=>{{ if(sb.value.length>=3) runSearch(sb.value); }});
}});
</script>
</body></html>
"""


def main() -> int:
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    OUT.write_text(render(g), encoding="utf-8")
    inlined = VENDOR.exists()
    print(f"Wrote {OUT.name} "
          f"({'vis-network inlined, works offline' if inlined else 'CDN fallback — needs internet'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
