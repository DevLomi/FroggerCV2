"""Summarize captured pothole photos and serve a Material 3 split-view HTML heatmap."""

import argparse
from collections import Counter
from datetime import datetime
from functools import partial
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from urllib.parse import quote
import webbrowser

ROOT = Path(__file__).resolve().parent
PHOTO_DIR = ROOT / "pothole_photos"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FILENAME_PATTERN = re.compile(r"^\[([^\]]+)\],\s*\[([^\]]+)\]$")


def parse_photo(path):
    """Extract optional (lat, lon) and timestamp from a pothole capture filename."""
    path = Path(path)
    lat = lon = captured_at = None
    match = FILENAME_PATTERN.match(path.stem)
    if match:
        loc_text, time_text = match.groups()
        if loc_text != "location-unavailable" and "_" in loc_text:
            try:
                raw_lat, raw_lon = map(float, loc_text.split("_", 1))
                if math.isfinite(raw_lat) and math.isfinite(raw_lon) and -90 <= raw_lat <= 90 and -180 <= raw_lon <= 180:
                    lat, lon = raw_lat, raw_lon
            except ValueError:
                pass
        try:
            captured_at = datetime.strptime(time_text, "%Y-%m-%d_%H-%M-%S.%f%z")
        except ValueError:
            pass
    return {"path": path, "name": path.name, "lat": lat, "lon": lon, "captured_at": captured_at}


def summarize_photos(photo_dir=PHOTO_DIR):
    """Aggregate counts, daily breakdown, and ~11m coordinate hotspots."""
    photo_dir = Path(photo_dir)
    files = sorted(
        p for p in photo_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ) if photo_dir.is_dir() else []
    records = [parse_photo(p) for p in files]
    located = [r for r in records if r["lat"] is not None]
    timestamps = sorted(r["captured_at"] for r in records if r["captured_at"] is not None)
    by_day = dict(sorted(Counter(t.strftime("%Y-%m-%d") for t in timestamps).items()))
    hotspots = Counter((round(r["lat"], 4), round(r["lon"], 4)) for r in located).most_common(10)
    return {
        "photo_dir": photo_dir,
        "total": len(records),
        "geolocated": len(located),
        "unavailable_location": len(records) - len(located),
        "first_capture": timestamps[0] if timestamps else None,
        "last_capture": timestamps[-1] if timestamps else None,
        "by_day": by_day,
        "hotspots": hotspots,
        "records": records,
    }


def format_summary(summary):
    """Format aggregated pothole photo statistics as plain text."""
    if summary["total"] == 0:
        return f"No pothole photos found in '{summary['photo_dir']}'."
    lines = [
        f"Pothole Summary ({summary['photo_dir']})",
        f"  Total photos      : {summary['total']}",
        f"  With GPS location : {summary['geolocated']}",
        f"  No GPS location   : {summary['unavailable_location']}",
    ]
    if summary["first_capture"] is not None:
        fmt = "%Y-%m-%d %H:%M:%S %z"
        lines.append(f"  First capture     : {summary['first_capture']:{fmt}}")
        lines.append(f"  Latest capture    : {summary['last_capture']:{fmt}}")
    if summary["by_day"]:
        lines.append("  Captures by day   :")
        for day, count in summary["by_day"].items():
            lines.append(f"    {day}: {count}")
    if summary["hotspots"]:
        lines.append("  Top hotspots (~11m):")
        for (lat, lon), count in summary["hotspots"]:
            lines.append(f"    {lat:.4f}, {lon:.4f}: {count} photo(s)")
    return "\n".join(lines)


def _photo_url(path, output_dir):
    try:
        return quote(path.resolve().relative_to(output_dir).as_posix())
    except ValueError:
        return path.resolve().as_uri()


def write_heatmap(summary, output_path=None):
    """Write a Material 3 50/50 stats-and-heatmap HTML dashboard."""
    located = [r for r in summary["records"] if r["lat"] is not None]
    if not located:
        return None
    output_path = Path(output_path) if output_path else Path(summary["photo_dir"]) / "heatmap.html"
    output_dir = output_path.resolve().parent

    grouped = {}
    for rec in located:
        key = (round(rec["lat"], 5), round(rec["lon"], 5))
        entry = grouped.setdefault(key, {"lat": key[0], "lon": key[1], "count": 0, "latest": "", "img": ""})
        entry["count"] += 1
        stamp = rec["captured_at"].strftime("%Y-%m-%d %H:%M:%S") if rec["captured_at"] else html.escape(rec["name"])
        entry["latest"] = stamp
        entry["img"] = _photo_url(rec["path"], output_dir)

    spots = sorted(grouped.values(), key=lambda s: s["count"], reverse=True)
    max_intensity = min(10, max(s["count"] for s in spots))
    heat_points = [[s["lat"], s["lon"], s["count"]] for s in spots]

    total = summary["total"]
    geolocated = summary["geolocated"]
    unlocated = summary["unavailable_location"]
    gps_pct = round(geolocated / total * 100) if total else 0

    if summary["first_capture"] is not None:
        span_text = (
            f"{summary['first_capture']:%Y-%m-%d %H:%M} to "
            f"{summary['last_capture']:%Y-%m-%d %H:%M} ({summary['last_capture']:%z})"
        )
    else:
        span_text = "No timestamp metadata"

    max_day = max(summary["by_day"].values(), default=1)
    day_rows = "\n".join(
        f"""<div class="bar-row">
          <span class="bar-label">{html.escape(day)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{round(count / max_day * 100)}%"></div></div>
          <span class="bar-value">{count}</span>
        </div>"""
        for day, count in summary["by_day"].items()
    ) or '<div class="empty-note">No dated captures</div>'

    recent_records = sorted(
        summary["records"],
        key=lambda r: (r["captured_at"] is not None, r["captured_at"] or r["name"]),
        reverse=True,
    )[:6]
    recent_cards = "\n".join(
        f"""<button type="button" class="thumb-card"
          data-img="{html.escape(_photo_url(r['path'], output_dir), quote=True)}"
          data-time="{html.escape(r['captured_at'].strftime('%Y-%m-%d %H:%M:%S') if r['captured_at'] else r['name'], quote=True)}"
          data-loc="{f"{r['lat']:.5f}, {r['lon']:.5f}" if r['lat'] is not None else "Location unavailable"}"
          data-lat="{r['lat'] if r['lat'] is not None else ''}"
          data-lon="{r['lon'] if r['lon'] is not None else ''}">
          <img src="{_photo_url(r['path'], output_dir)}" alt="Pothole capture" loading="lazy">
          <div class="thumb-meta">
            <div class="thumb-time">{r['captured_at'].strftime('%Y-%m-%d %H:%M:%S') if r['captured_at'] else html.escape(r['name'])}</div>
            <div class="thumb-loc">{f"{r['lat']:.5f}, {r['lon']:.5f}" if r['lat'] is not None else "Location unavailable"}</div>
          </div>
        </button>"""
        for r in recent_records
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="strict-origin-when-cross-origin">
<title>Pothole Summary &amp; Heatmap</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root {{
    --surface: #f4f7f5;
    --surface-card: #ffffff;
    --surface-tonal: #e9efec;
    --primary: #006a60;
    --primary-container: #d3ebe6;
    --on-primary-container: #00201c;
    --alert-container: #f9dedc;
    --on-alert-container: #410e0b;
    --alert: #b3261e;
    --on-surface: #191c1b;
    --on-surface-variant: #444c49;
    --outline-variant: #d8e0dc;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    height: 100%;
    margin: 0;
    font-family: "Segoe UI", Roboto, system-ui, -apple-system, sans-serif;
    background: var(--surface);
    color: var(--on-surface);
  }}
  .layout {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    height: 100vh;
    width: 100vw;
  }}
  .stats-pane {{
    padding: 28px 32px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
    border-right: 1px solid var(--outline-variant);
  }}
  .header-eyebrow {{
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 4px;
  }}
  .header-title {{
    margin: 0;
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }}
  .header-sub {{
    margin: 6px 0 0;
    font-size: 13px;
    color: var(--on-surface-variant);
  }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
  }}
  .metric-card {{
    background: var(--surface-card);
    border: 1px solid var(--outline-variant);
    border-radius: 16px;
    padding: 16px 18px;
  }}
  .metric-card.primary {{
    background: var(--primary-container);
    border-color: var(--primary-container);
    color: var(--on-primary-container);
  }}
  .metric-label {{
    font-size: 13px;
    font-weight: 500;
    color: var(--on-surface-variant);
  }}
  .metric-card.primary .metric-label,
  .metric-card.primary .metric-note {{
    color: var(--on-primary-container);
    opacity: 0.85;
  }}
  .metric-value {{
    font-size: 30px;
    font-weight: 600;
    margin: 6px 0 2px;
    line-height: 1.1;
  }}
  .metric-note {{
    font-size: 12px;
    color: var(--on-surface-variant);
  }}
  .section-card {{
    background: var(--surface-card);
    border: 1px solid var(--outline-variant);
    border-radius: 16px;
    padding: 18px 20px;
  }}
  .section-title {{
    margin: 0 0 14px;
    font-size: 15px;
    font-weight: 600;
    color: var(--on-surface);
  }}
  .bar-row {{
    display: grid;
    grid-template-columns: 92px minmax(0, 1fr) 36px;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    font-size: 13px;
  }}
  .bar-row:last-child {{ margin-bottom: 0; }}
  .bar-label {{ color: var(--on-surface-variant); font-variant-numeric: tabular-nums; }}
  .bar-track {{
    height: 10px;
    background: var(--surface-tonal);
    border-radius: 999px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    background: var(--primary);
    border-radius: 999px;
  }}
  .bar-value {{
    text-align: right;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .hotspot-list {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .hotspot-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid transparent;
    border-radius: 12px;
    cursor: pointer;
    text-align: left;
    font: inherit;
    color: inherit;
    width: 100%;
  }}
  .hotspot-item:hover {{
    background: var(--surface-tonal);
    border-color: var(--outline-variant);
  }}
  .hotspot-coord {{
    font-size: 14px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .hotspot-time {{
    font-size: 12px;
    color: var(--on-surface-variant);
    margin-top: 2px;
  }}
  .pill {{
    background: var(--alert-container);
    color: var(--on-alert-container);
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    white-space: nowrap;
  }}
  .thumb-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
  }}
  .thumb-card {{
    border: 1px solid var(--outline-variant);
    border-radius: 12px;
    overflow: hidden;
    color: inherit;
    background: var(--surface);
    display: block;
    padding: 0;
    text-align: left;
    font: inherit;
    cursor: pointer;
    width: 100%;
  }}
  .thumb-card:hover {{ border-color: var(--primary); }}
  .thumb-card img {{
    width: 100%;
    height: 92px;
    object-fit: cover;
    display: block;
    background: var(--surface-tonal);
  }}
  .thumb-meta {{ padding: 8px 10px; }}
  .thumb-time {{ font-size: 11px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .thumb-loc {{ font-size: 11px; color: var(--on-surface-variant); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .empty-note {{ font-size: 13px; color: var(--on-surface-variant); }}
  #map {{ height: 100%; width: 100%; background: var(--surface-tonal); }}
  .popup-img {{ width: 220px; max-height: 150px; object-fit: cover; border-radius: 8px; margin-top: 8px; display: block; }}
  .photo-modal {{
    border: 1px solid var(--outline-variant);
    border-radius: 24px;
    padding: 0;
    background: var(--surface-card);
    color: var(--on-surface);
    max-width: min(760px, 92vw);
    width: 100%;
  }}
  .photo-modal::backdrop {{
    background: rgba(25, 28, 27, 0.5);
  }}
  .modal-body {{
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .modal-img {{
    width: 100%;
    max-height: 68vh;
    object-fit: contain;
    border-radius: 14px;
    background: var(--surface-tonal);
    display: block;
  }}
  .modal-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .modal-title {{ font-size: 15px; font-weight: 600; }}
  .modal-sub {{ font-size: 13px; color: var(--on-surface-variant); margin-top: 2px; }}
  .modal-actions {{ display: flex; gap: 8px; }}
  .btn {{
    border: none;
    border-radius: 999px;
    padding: 8px 16px;
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    background: var(--surface-tonal);
    color: var(--on-surface);
  }}
  .btn.tonal {{
    background: var(--primary-container);
    color: var(--on-primary-container);
  }}
  .btn:hover {{ opacity: 0.9; }}
  @media (max-width: 900px) {{
    .layout {{ grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }}
    .stats-pane {{ border-right: none; border-bottom: 1px solid var(--outline-variant); padding: 20px; }}
    .thumb-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  }}
</style>
</head>
<body>
<div class="layout">
  <aside class="stats-pane">
    <header>
      <div class="header-eyebrow">FroggerCV Analytics</div>
      <h1 class="header-title">Pothole Summary</h1>
      <p class="header-sub">{html.escape(span_text)}</p>
    </header>

    <section class="metrics-grid">
      <div class="metric-card primary">
        <div class="metric-label">Total Potholes</div>
        <div class="metric-value">{total}</div>
        <div class="metric-note">Captured frames</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">GPS Mapped</div>
        <div class="metric-value">{geolocated}</div>
        <div class="metric-note">{gps_pct}% location coverage</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Unique Hotspots</div>
        <div class="metric-value">{len(spots)}</div>
        <div class="metric-note">Distinct coordinates</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Unlocated</div>
        <div class="metric-value">{unlocated}</div>
        <div class="metric-note">No GPS fix</div>
      </div>
    </section>

    <section class="section-card">
      <h2 class="section-title">Captures by Day</h2>
      {day_rows}
    </section>

    <section class="section-card">
      <h2 class="section-title">Hotspots (Click to Focus Map)</h2>
      <div id="hotspot-list" class="hotspot-list"></div>
    </section>

    <section class="section-card">
      <h2 class="section-title">Recent Captures</h2>
      <div class="thumb-grid">
        {recent_cards}
      </div>
    </section>
  </aside>

  <main id="map"></main>
</div>

<dialog id="photo-modal" class="photo-modal">
  <div class="modal-body">
    <img id="modal-img" class="modal-img" src="" alt="Pothole capture">
    <div class="modal-footer">
      <div>
        <div id="modal-time" class="modal-title"></div>
        <div id="modal-loc" class="modal-sub"></div>
      </div>
      <div class="modal-actions">
        <button type="button" id="modal-locate" class="btn tonal">Focus on Map</button>
        <button type="button" id="modal-close" class="btn">Close</button>
      </div>
    </div>
  </div>
</dialog>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<script>
const heatPoints = {json.dumps(heat_points)};
const spots = {json.dumps(spots)};
const listEl = document.getElementById("hotspot-list");
const modal = document.getElementById("photo-modal");
const modalImg = document.getElementById("modal-img");
const modalTime = document.getElementById("modal-time");
const modalLoc = document.getElementById("modal-loc");
const modalLocate = document.getElementById("modal-locate");
let leafletMap = null;
let activeCoords = null;

document.querySelectorAll(".thumb-card").forEach((card) => {{
  card.addEventListener("click", () => {{
    modalImg.src = card.dataset.img;
    modalTime.textContent = card.dataset.time;
    modalLoc.textContent = card.dataset.loc;
    if (card.dataset.lat && card.dataset.lon) {{
      activeCoords = [parseFloat(card.dataset.lat), parseFloat(card.dataset.lon)];
      modalLocate.hidden = false;
    }} else {{
      activeCoords = null;
      modalLocate.hidden = true;
    }}
    modal.showModal();
  }});
}});

document.getElementById("modal-close").addEventListener("click", () => modal.close());
modal.addEventListener("click", (e) => {{
  if (e.target === modal) modal.close();
}});
modalLocate.addEventListener("click", () => {{
  if (leafletMap && activeCoords) leafletMap.setView(activeCoords, 17);
  modal.close();
}});

if (typeof L === "undefined") {{
  document.getElementById("map").innerHTML = "<p style='padding:2rem'>Internet connection required to load map tiles.</p>";
}} else {{
  const map = L.map("map");
  leafletMap = map;
  const isLocalFile = location.protocol === "file:";
  const tileUrl = isLocalFile
    ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}"
    : "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png";
  const tileAttr = isLocalFile
    ? "Tiles &copy; Esri"
    : "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors";
  L.tileLayer(tileUrl, {{ maxZoom: 19, attribution: tileAttr }}).addTo(map);

  if (typeof L.heatLayer === "function") {{
    L.heatLayer(heatPoints, {{
      radius: 28,
      blur: 18,
      minOpacity: 0.45,
      max: {max_intensity},
      gradient: {{ 0.25: "#14b8a6", 0.55: "#eab308", 0.8: "#f97316", 1.0: "#b3261e" }}
    }}).addTo(map);
  }}

  const bounds = [];
  const markers = [];
  spots.slice(0, 15).forEach((s, idx) => {{
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "hotspot-item";
    btn.innerHTML = `<div><div class="hotspot-coord">${{s.lat.toFixed(5)}}, ${{s.lon.toFixed(5)}}</div>`
      + `<div class="hotspot-time">Latest: ${{s.latest}}</div></div>`
      + `<span class="pill">${{s.count}} ${{s.count === 1 ? "photo" : "photos"}}</span>`;
    btn.addEventListener("click", () => {{
      map.setView([s.lat, s.lon], 17);
      if (markers[idx]) markers[idx].openPopup();
    }});
    listEl.appendChild(btn);
  }});

  spots.forEach((s) => {{
    bounds.push([s.lat, s.lon]);
    const mapsUrl = `https://www.google.com/maps?q=${{s.lat}},${{s.lon}}`;
    const popup = `<b>${{s.lat.toFixed(5)}}, ${{s.lon.toFixed(5)}}</b><br>`
      + `Detections: ${{s.count}}<br>Latest: ${{s.latest}}<br>`
      + `<a href="${{mapsUrl}}" target="_blank" rel="noopener">Open in Google Maps</a>`
      + (s.img ? `<img class="popup-img" src="${{s.img}}" alt="Pothole">` : "");
    const marker = L.circleMarker([s.lat, s.lon], {{
      radius: Math.min(14, 6 + s.count),
      color: "#ffffff",
      weight: 2,
      fillColor: "#b3261e",
      fillOpacity: 0.85
    }}).bindPopup(popup).addTo(map);
    markers.push(marker);
  }});
  map.fitBounds(bounds, {{ padding: [48, 48], maxZoom: 16 }});
}}
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


def serve_heatmap(heatmap_path):
    """Serve the heatmap directory over localhost so OpenStreetMap receives a valid HTTP Referer."""
    heatmap_path = Path(heatmap_path).resolve()

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = partial(QuietHandler, directory=str(heatmap_path.parent))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        url = f"http://localhost:{server.server_port}/{quote(heatmap_path.name)}"
        print(f"Opening {url} (press Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=PHOTO_DIR, help="Directory of pothole photos.")
    parser.add_argument("--heatmap", type=Path, default=None, help="Output HTML heatmap file path.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the heatmap in a browser.")
    args = parser.parse_args()

    summary = summarize_photos(args.dir)
    print(format_summary(summary))
    if summary["geolocated"] > 0:
        heatmap_path = write_heatmap(summary, args.heatmap)
        print(f"Heatmap saved to '{heatmap_path}'")
        if not args.no_open:
            serve_heatmap(heatmap_path)
    elif summary["total"] > 0:
        print("Heatmap skipped: no photos contain GPS coordinates.")


if __name__ == "__main__":
    main()
