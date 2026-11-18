"""Folium map generation (reused by the EDA scripts, Member 3's decoder comparison and Member 4's UI).

make_trip_map(trip, ...) draws, as separate toggleable layers:
    raw GPS observations (blue dots), road network near the trip (grey), reference route (green),
    predicted route (red, optional), with a legend and a layer control.
`predicted_arcs` is a sequence of arc ids (one per GPS observation OR the ordered route); consecutive
repeats are collapsed. Never pass evaluator ground truth into a map that is shown in the student UI.
"""
import numpy as np


def _arc_segments(trip, arc_ids):
    n = trip.nodes
    lon, lat = n["lon"].to_numpy(), n["lat"].to_numpy()
    u = trip.arcs["u"].to_numpy()[arc_ids]
    v = trip.arcs["v"].to_numpy()[arc_ids]
    return [[[float(lat[a]), float(lon[a])], [float(lat[b]), float(lon[b])]] for a, b in zip(u, v)]


def _dedupe(seq):
    seq = np.asarray(seq, dtype=np.int64)
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]


def make_trip_map(trip, out_html=None, predicted_arcs=None, max_gps=3000, max_network_arcs=40000,
                  margin_deg=0.003, show_network=False):
    import folium  # imported lazily so the rest of the package works without folium

    trk = trip.track
    lat, lon = trk["lat"].to_numpy(), trk["lon"].to_numpy()
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])

    # --- raw GPS ---
    g_gps = folium.FeatureGroup(name="Raw GPS observations (blue)", show=True)
    step = max(1, int(np.ceil(len(trk) / max_gps)))
    for i in range(0, len(trk), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2, color="#1f77b4", fill=True,
                            fill_opacity=0.8, weight=1,
                            tooltip=f"obs {i}  t={trk['t'].iloc[i]:.0f}s").add_to(g_gps)
    folium.Marker([float(lat[0]), float(lon[0])], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(g_gps)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="End",
                  icon=folium.Icon(color="black", icon="stop")).add_to(g_gps)
    g_gps.add_to(m)

    # --- road network near the trip ---
    nodes, arcs = trip.nodes, trip.arcs
    nlon, nlat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()
    lo0, lo1, la0, la1 = lon.min() - margin_deg, lon.max() + margin_deg, lat.min() - margin_deg, lat.max() + margin_deg
    keep = ((np.maximum(nlon[u], nlon[v]) >= lo0) & (np.minimum(nlon[u], nlon[v]) <= lo1)
            & (np.maximum(nlat[u], nlat[v]) >= la0) & (np.minimum(nlat[u], nlat[v]) <= la1))
    ids = np.flatnonzero(keep)[:max_network_arcs]
    g_net = folium.FeatureGroup(name=f"Road network ({len(ids)} arcs, grey)", show=show_network)
    if len(ids):
        folium.PolyLine(_arc_segments(trip, ids), color="#777777", weight=1.5, opacity=0.6).add_to(g_net)
    g_net.add_to(m)

    # --- reference route ---
    if trip.route is not None and len(trip.route):
        g_ref = folium.FeatureGroup(name="Reference route (green)", show=True)
        folium.PolyLine(_arc_segments(trip, trip.route), color="#2ca02c", weight=5, opacity=0.8,
                        tooltip="Reference route").add_to(g_ref)
        g_ref.add_to(m)

    # --- predicted route ---
    if predicted_arcs is not None and len(predicted_arcs):
        g_pred = folium.FeatureGroup(name="Predicted route (red)", show=True)
        folium.PolyLine(_arc_segments(trip, _dedupe(predicted_arcs)), color="#d62728", weight=3,
                        opacity=0.9, tooltip="Predicted route").add_to(g_pred)
        g_pred.add_to(m)

    legend = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
         padding: 8px 12px; border: 1px solid #888; border-radius: 4px; font-size: 13px;">
      <b>Trip {tid} ({part})</b><br>
      <span style="color:#1f77b4">&#9679;</span> Raw GPS observations<br>
      <span style="color:#777777">&#9644;</span> Road network<br>
      <span style="color:#2ca02c">&#9644;</span> Reference route<br>
      <span style="color:#d62728">&#9644;</span> Predicted route
    </div>""".format(tid=trip.trip_id, part=trip.partition)
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    if out_html:
        m.save(out_html)
    return m
