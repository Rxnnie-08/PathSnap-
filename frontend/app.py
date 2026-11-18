"""
PathSnap UI (Laptop 2).  Save as frontend/app.py

Install:  pip install streamlit streamlit-folium folium requests
Run:      streamlit run app.py
Then enter Laptop 1's address in the sidebar, e.g. http://192.168.1.10:8000
The UI holds no data and no labels; everything comes from the API.
"""
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: Map-Matching Evaluation")

API = st.sidebar.text_input("API URL", "http://127.0.0.1:8000").rstrip("/")
show_gps = st.sidebar.checkbox("Show GPS observations", True)
gps_step = st.sidebar.slider("Plot every n-th GPS point", 1, 10, 3)

try:
    health = requests.get(f"{API}/health", timeout=10).json()
    trips = requests.get(f"{API}/trips", timeout=10).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach API at {API}: {e}")
    st.stop()

st.sidebar.success(f"API ok | predictor: {health.get('predictor')} | trips: {len(trips)}")
if not trips:
    st.warning("The API reports no trips. Check DATA_DIR on the API laptop.")
    st.stop()

trip = st.selectbox("Evaluation trip", trips)
if st.button("Run prediction", type="primary"):
    try:
        with st.spinner("Running the pipeline on the server..."):
            r = requests.post(f"{API}/predict", json={"trip_id": trip}, timeout=300)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            st.session_state["res"] = r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")

res = st.session_state.get("res")
if res:
    m = res["metrics"]
    fmt = lambda v: "N/A" if v is None else f"{v:.3f}"
    st.subheader(f"Trip {res['trip_id']}  |  predictor: {res['predictor']}")
    c = st.columns(5)
    c[0].metric("Arc F1", fmt(m.get("arc_f1")))
    c[1].metric("Arc precision", fmt(m.get("arc_precision")))
    c[2].metric("Arc recall", fmt(m.get("arc_recall")))
    c[3].metric("Route continuity", fmt(m.get("route_continuity")))
    c[4].metric("Latency", f"{res['latency_ms']} ms")
    st.caption(f"{res['n_points']} GPS points | {len(res['route_arcs'])} predicted arcs")

    gps = res["gps"]
    if not gps:
        st.warning("No valid GPS points returned for this trip.")
        st.stop()
    fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=15, control_scale=True)
    route_layer = folium.FeatureGroup("Predicted route (red)", show=True)
    for seg in res["segments"]:
        folium.PolyLine(seg, color="red", weight=5, opacity=0.8).add_to(route_layer)
    route_layer.add_to(fmap)
    if show_gps:
        gps_layer = folium.FeatureGroup("GPS observations (blue)", show=True)
        for p in gps[::gps_step]:
            folium.CircleMarker(p, radius=2, color="blue", fill=True).add_to(gps_layer)
        gps_layer.add_to(fmap)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)],
                     [max(p[0] for p in gps), max(p[1] for p in gps)]])
    st_folium(fmap, height=620, use_container_width=True)