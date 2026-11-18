
import { useEffect, useMemo, useState } from "react";
import {
  Activity, Map, Layers, Route, Settings, Search, Bell,
  ChevronDown, Crosshair, Radio, Database, Cpu, Network,
  Clock3, ChartNoAxesCombined, Zap, Menu, SlidersHorizontal,
  ArrowUpRight, LocateFixed, Workflow, Server, Gauge,
  CircleHelp, PanelLeftClose, Signal
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

function StatusDot({ online }) {
  return <span className={`status-dot ${online ? "online" : "offline"}`} />;
}

function PanelTitle({ children, action }) {
  return (
    <div className="panel-heading">
      <span className="panel-title">{children}</span>
      {action}
    </div>
  );
}

function App() {
  const [apiOnline, setApiOnline] = useState(false);
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [activeTab, setActiveTab] = useState("map");

  useEffect(() => {
    let mounted = true;

    async function checkBackend() {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (mounted) setApiOnline(response.ok);
      } catch {
        if (mounted) setApiOnline(false);
      }
    }

    async function fetchTrips() {
      try {
        const response = await fetch(`${API_URL}/trips/`);
        if (!response.ok) return;
        const data = await response.json();
        if (mounted) setTrips(data.trips || []);
      } catch {
        if (mounted) setTrips([]);
      }
    }

    checkBackend();
    fetchTrips();

    const interval = setInterval(checkBackend, 10000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const filteredTrips = useMemo(() => {
    return trips.filter((trip) =>
      trip.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [trips, searchTerm]);

  async function reconstructRoute() {
    if (!selectedTrip) return;

    setLoading(true);
    setMessage("");
    setPrediction(null);

    try {
      const response = await fetch(
        `${API_URL}/predict/${encodeURIComponent(selectedTrip)}`,
        { method: "POST" }
      );

      const data = await response.json();

      if (!response.ok) {
        setMessage(data.detail || `Request failed: ${response.status}`);
      } else {
        setPrediction(data);
        setMessage("Route reconstruction completed.");
      }
    } catch {
      setMessage("Unable to connect to the PathSnap API.");
    } finally {
      setLoading(false);
    }
  }

  const accuracy = prediction?.evaluation?.accuracy;
  const macroF1 = prediction?.evaluation?.macro_f1;
  const logLoss = prediction?.evaluation?.log_loss;

  return (
    <main className="app-shell">

      {/* HEADER */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">◈</div>
          <div>
            <div className="brand-name">PATHSNAP</div>
            <div className="brand-subtitle">GEOSPATIAL INTELLIGENCE</div>
          </div>
        </div>

        <div className="header-title">
          <span>/</span> CONTROL TOWER
        </div>

        <div className="header-actions">
          <div className="connection-indicator">
            <StatusDot online={apiOnline} />
            {apiOnline ? "SYSTEM CONNECTED" : "API OFFLINE"}
          </div>

          <button className="icon-button" title="Notifications">
            <Bell size={16} />
          </button>
          <button className="icon-button" title="Settings">
            <Settings size={16} />
          </button>
          <div className="user-avatar">CS</div>
        </div>
      </header>

      <div className="dashboard-layout">

        {/* NAVIGATION RAIL */}
        <nav className="nav-rail">
          <div className="rail-brand">PS</div>

          <button
            className={`nav-item ${activeTab === "map" ? "active" : ""}`}
            title="Overview"
            onClick={() => setActiveTab("map")}
          >
            <Layers size={19} />
          </button>

          <button
            className={`nav-item ${activeTab === "routes" ? "active" : ""}`}
            title="Routes"
            onClick={() => setActiveTab("routes")}
          >
            <Route size={19} />
          </button>

          <button
            className={`nav-item ${activeTab === "analytics" ? "active" : ""}`}
            title="Analytics"
            onClick={() => setActiveTab("analytics")}
          >
            <ChartNoAxesCombined size={19} />
          </button>

          <button className="nav-item" title="System">
            <Activity size={19} />
          </button>

          <div className="rail-spacer" />

          <button className="nav-item" title="Settings">
            <Settings size={19} />
          </button>
        </nav>

        {/* LEFT OPERATIONS PANEL */}
        <aside className="operations-panel">

          <section>
            <PanelTitle>OVERVIEW</PanelTitle>
            <div className="section-caption">QUICK ACTIONS</div>

            <div className="quick-actions">
              <button
                className="quick-action"
                onClick={() => document.getElementById("trip-search")?.focus()}
              >
                <span className="quick-icon cyan-bg"><Route size={17} /></span>
                <span>Trips</span>
              </button>

              <button
                className="quick-action"
                onClick={() => setActiveTab("analytics")}
              >
                <span className="quick-icon purple-bg"><Cpu size={17} /></span>
                <span>Models</span>
              </button>

              <button
                className="quick-action"
                onClick={() => setActiveTab("map")}
              >
                <span className="quick-icon green-bg"><Network size={17} /></span>
                <span>Network</span>
              </button>

              <button
                className="quick-action"
                onClick={() => setMessage("Dataset status is shown below.")}
              >
                <span className="quick-icon blue-bg"><Database size={17} /></span>
                <span>Data</span>
              </button>
            </div>
          </section>

          <section className="monitor-section">
            <PanelTitle>SYSTEM MONITOR</PanelTitle>

            <div className="monitor-row">
              <span><Radio size={13} /> API SERVICE</span>
              <span className={apiOnline ? "green-text" : "red-text"}>
                <StatusDot online={apiOnline} />
                {apiOnline ? "ONLINE" : "OFFLINE"}
              </span>
            </div>

            <div className="monitor-row">
              <span><Cpu size={13} /> ML ENGINE</span>
              <span className="muted-text">PENDING</span>
            </div>

            <div className="monitor-row">
              <span><Workflow size={13} /> ROUTE DECODER</span>
              <span className="muted-text">PENDING</span>
            </div>
          </section>

          <section className="trip-section">
            <div className="section-heading-row">
              <PanelTitle>EVALUATION TRIPS</PanelTitle>
              <span className="count-label">{trips.length}</span>
            </div>

            <div className="trip-summary">
              <div>
                <div className="section-caption">AVAILABLE TRIPS</div>
                <div className="trip-count">{trips.length || "—"}</div>
              </div>
              <Route size={22} className="muted-icon" />
            </div>

            <div className="trip-search">
              <Search size={14} />
              <input
                id="trip-search"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search trips"
              />
            </div>

            {filteredTrips.length > 0 ? (
              <div className="trip-list">
                {filteredTrips.map((trip) => (
                  <button
                    key={trip}
                    className={`trip-item ${selectedTrip === trip ? "selected" : ""}`}
                    onClick={() => setSelectedTrip(trip)}
                  >
                    <Route size={14} />
                    <span>{trip}</span>
                    <span className="trip-arrow">›</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <Database size={21} />
                <span>
                  {trips.length ? "No matching trips" : "Dataset not connected"}
                </span>
                <small>
                  {trips.length
                    ? "Try another search"
                    : "Waiting for evaluation trip data"}
                </small>
              </div>
            )}
          </section>

          <div className="sidebar-footer">
            <span>PATHSNAP BUILD 1.0</span>
            <span>MEMBER 4 · BACKEND & UI</span>
          </div>
        </aside>

        {/* CENTRAL WORKSPACE */}
        <section className="workspace">

          <div className="workspace-toolbar">
            <button
              className={`workspace-tab ${activeTab === "map" ? "active-tab" : ""}`}
              onClick={() => setActiveTab("map")}
            >
              <Map size={16} />
              <span>
                <strong>Node Graph View</strong>
                <small>Geospatial</small>
              </span>
            </button>

            <button
              className={`workspace-tab ${activeTab === "analytics" ? "active-tab" : ""}`}
              onClick={() => setActiveTab("analytics")}
            >
              <ChartNoAxesCombined size={16} />
              <span>
                <strong>Visualizations</strong>
                <small>Route analytics</small>
              </span>
            </button>

            <button
              className={`workspace-tab ${activeTab === "routes" ? "active-tab" : ""}`}
              onClick={() => setActiveTab("routes")}
            >
              <Zap size={16} />
              <span>
                <strong>Actions</strong>
                <small>Reconstruction</small>
              </span>
            </button>

            <div className="toolbar-spacer" />

            <button className="icon-button" title="Map controls">
              <Search size={16} />
            </button>
          </div>

          <div className="map-container">
            <div className="map-top-overlay">
              <span className="map-label">
                {activeTab === "map"
                  ? "GEOSPATIAL NETWORK"
                  : activeTab === "analytics"
                    ? "ROUTE ANALYTICS"
                    : "ROUTE RECONSTRUCTION"}
              </span>
              <span className="map-live">
                <StatusDot online={apiOnline} />
                {apiOnline ? "API ONLINE" : "API OFFLINE"}
              </span>
            </div>

            <div className="map-placeholder">
              <div className="map-center">
                <div className="map-crosshair">
                  <Crosshair size={39} />
                </div>

                <h3>
                  {prediction
                    ? "ROUTE RESPONSE RECEIVED"
                    : "GEOSPATIAL VIEWPORT"}
                </h3>

                <p>
                  {prediction
                    ? `Reconstruction response for ${prediction.trip_id}`
                    : "Awaiting GPS observations and road-network geometry"}
                </p>

                <span>
                  {prediction
                    ? "MAP RENDERING AWAITS COORDINATE DATA"
                    : "MAP DATA NOT CONNECTED"}
                </span>
              </div>
            </div>

            <div className="map-bottom-overlay">
              <span><i className="legend-dot cyan-dot" /> GPS OBSERVATIONS</span>
              <span><i className="legend-dot purple-dot" /> ROAD NETWORK</span>
              <span><i className="legend-dot red-dot" /> PREDICTED ROUTE</span>
            </div>
          </div>

          {/* ROUTE TIMELINE */}
          <div className="timeline-panel">
            <div className="timeline-header">
              <PanelTitle>ROUTE SEQUENCE / TIMELINE</PanelTitle>
              <span className="muted-text">SEQUENCE VIEW</span>
            </div>

            {prediction?.predicted_route?.length > 0 ? (
              <div className="sequence-list">
                {prediction.predicted_route.map((segment, index) => (
                  <span className="sequence-segment" key={`${segment}-${index}`}>
                    {segment}
                  </span>
                ))}
              </div>
            ) : (
              <div className="timeline-empty">
                <Clock3 size={18} />
                <span>Route reconstruction sequence will appear here</span>
              </div>
            )}
          </div>
        </section>

        {/* RIGHT DETAILS PANEL */}
        <aside className="details-panel">

          <div className="details-header">
            <PanelTitle>DETAILS / INSPECTOR</PanelTitle>
            <button className="icon-button" title="Inspector search">
              <Search size={15} />
            </button>
          </div>

          <section className="detail-card">
            <PanelTitle>TRIP SELECTION</PanelTitle>

            <label className="field-label">EVALUATION TRIP</label>

            <div className="select-wrapper">
              <select
                value={selectedTrip}
                onChange={(e) => setSelectedTrip(e.target.value)}
              >
                <option value="">Select a trip</option>
                {trips.map((trip) => (
                  <option key={trip} value={trip}>{trip}</option>
                ))}
              </select>
              <ChevronDown size={14} />
            </div>

            <button
              className="reconstruct-button"
              disabled={!selectedTrip || loading}
              onClick={reconstructRoute}
            >
              {loading ? "PROCESSING..." : "↗  RECONSTRUCT ROUTE"}
            </button>

            {message && <div className="feedback-message">{message}</div>}
          </section>

          <section className="detail-card">
            <PanelTitle>SYSTEM INTELLIGENCE</PanelTitle>

            <div className="detail-row">
              <span>API Service</span>
              <strong className={apiOnline ? "green-text" : "red-text"}>
                {apiOnline ? "Connected" : "Offline"}
              </strong>
            </div>

            <div className="detail-row">
              <span>ML Model</span>
              <strong className="muted-text">Pending</strong>
            </div>

            <div className="detail-row">
              <span>HMM Decoder</span>
              <strong className="muted-text">Pending</strong>
            </div>
          </section>

          <section className="detail-card">
            <PanelTitle>MODEL PERFORMANCE</PanelTitle>

            <div className="performance-placeholder">
              <div className="performance-ring">
                <span>
                  {typeof accuracy === "number"
                    ? `${(accuracy * 100).toFixed(1)}%`
                    : "—"}
                </span>
                <small>ACCURACY</small>
              </div>

              <div className="performance-legend">
                <span><i className="legend-dot purple-dot" /> Accuracy</span>
                <span><i className="legend-dot cyan-dot" /> Macro F1</span>
                <span><i className="legend-dot green-dot" /> Log Loss</span>
              </div>
            </div>

            <div className="detail-row">
              <span>Macro F1</span>
              <strong>
                {typeof macroF1 === "number" ? macroF1.toFixed(4) : "—"}
              </strong>
            </div>

            <div className="detail-row">
              <span>Log Loss</span>
              <strong>
                {typeof logLoss === "number" ? logLoss.toFixed(4) : "—"}
              </strong>
            </div>

            {!prediction && (
              <small className="muted-text">
                Actual evaluation metrics will appear after integration.
              </small>
            )}
          </section>

          <section className="detail-card">
            <PanelTitle>PROCESSING DETAILS</PanelTitle>

            <div className="detail-row">
              <span>Selected trip</span>
              <strong>{selectedTrip || "—"}</strong>
            </div>

            <div className="detail-row">
              <span>Processing time</span>
              <strong>
                {typeof prediction?.processing_time_seconds === "number"
                  ? `${prediction.processing_time_seconds.toFixed(3)} s`
                  : "—"}
              </strong>
            </div>

            <div className="detail-row">
              <span>Predicted segments</span>
              <strong>
                {prediction?.predicted_route?.length ?? "—"}
              </strong>
            </div>
          </section>

          {prediction && (
            <section className="detail-card">
              <PanelTitle>PREDICTION RESPONSE</PanelTitle>
              <pre className="prediction-json">
                {JSON.stringify(prediction, null, 2)}
              </pre>
            </section>
          )}
        </aside>
      </div>
    </main>
  );
}

export default App;
