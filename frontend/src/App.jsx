import AnalyticsDashboard from "./AnalyticsDashboard";

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
        setMessage(
  data.status === "candidate_top1" ? "Candidate predictions generated successfully. Temporal route decoding is pending."  : "Prediction completed successfully."
);
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
          <div className="brand-mark">PS</div>
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
            onClick={() => setActiveTab("models")}
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
                onClick={() => setActiveTab("trips")}
              >
                <span className="quick-icon cyan-bg"><Route size={17} /></span>
                <span>Trips</span>
              </button>

              <button
                className="quick-action"
                onClick={() => setActiveTab("models")}
              >
                <span className="quick-icon purple-bg"><Cpu size={17} /></span>
                <span>Models</span>
              </button>

              <button
                className="quick-action"
                onClick={() => setActiveTab("network")}
              >
                <span className="quick-icon green-bg"><Network size={17} /></span>
                <span>Network</span>
              </button>

              <button
                className="quick-action"
                onClick={() => setActiveTab("data")}
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
              <span className="green-text">LOADED</span>
            </div>

            <div className="monitor-row">
              <span><Workflow size={13} /> ROUTE DECODER</span>
              <span className="green-text">LOADED</span>
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
                <div className="trip-count">{trips.length || "N/A"}</div>
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
                    <span className="trip-arrow">&gt;</span>
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
            <span>MEMBER 4 | BACKEND & UI</span>
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

          {["trips", "models", "network", "data", "analytics"].includes(activeTab) ? (
            <div className="map-container info-view">
              <div className="info-view-header">
                <span className="map-label">PATHSNAP / CONTROL TOWER</span>
                <span className="map-live">
                  <StatusDot online={apiOnline} />
                  {apiOnline ? "API ONLINE" : "API OFFLINE"}
                </span>
              </div>

              {activeTab === "trips" && (
                <div className="info-content">
                  <h2>Trip Explorer</h2>
                  <p>Browse and select available student-test trips.</p>
                  <div className="info-stat-grid">
                    <div className="info-stat">
                      <strong>{trips.length}</strong>
                      <span>Available trips</span>
                    </div>
                    <div className="info-stat">
                      <strong>{selectedTrip || "None"}</strong>
                      <span>Selected trip</span>
                    </div>
                  </div>
                  <h3>Available trips</h3>
                  <div className="info-trip-list">
                    {trips.map((trip) => (
                      <button
                        key={trip}
                        className={`trip-item ${selectedTrip === trip ? "selected" : ""}`}
                        onClick={() => setSelectedTrip(trip)}
                      >
                        <Route size={15} />
                        <span>{trip}</span>
                        <span className="trip-arrow">&gt;</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === "models" && (
                <div className="info-content">
                  <h2>Machine Learning Models</h2>
                  <p>Classifier currently integrated with the route reconstruction pipeline.</p>
                  <div className="info-stat-grid">
                    <div className="info-stat">
                      <strong>Random Forest</strong>
                      <span>Final classifier</span>
                    </div>
                    <div className="info-stat">
                      <strong>25</strong>
                      <span>Input features</span>
                    </div>
                    <div className="info-stat">
                      <strong>2</strong>
                      <span>Output classes: 0 and 1</span>
                    </div>
                  </div>
                  <h3>Inference pipeline</h3>
                  <div className="info-pipeline">
                    <span>25-feature candidate input</span>
                    <span>↓</span>
                    <span>Random Forest classifier</span>
                    <span>↓</span>
                    <span>Probability scores</span>
                    <span>↓</span>
                    <span>HMM / Viterbi decoder</span>
                  </div>
                  <p className="info-note">Model artifact: artifacts/models/best_classifier.joblib</p>
                  <div className="model-theory">
                    <h3>TECHNICAL OVERVIEW</h3>

                    <section className="theory-section">
                      <h4>01 / What is Map Matching?</h4>
                      <p>
                        GPS devices record geographic positions, but these observations
                        may contain noise and may not fall exactly on a road. Map matching
                        identifies the most plausible road segments corresponding to a
                        sequence of GPS observations.
                      </p>
                    </section>

                    <section className="theory-section">
                      <h4>02 / Candidate Generation</h4>
                      <p>
                        For each GPS observation, PathSnap identifies nearby candidate
                        road segments. Each candidate is represented using engineered
                        information about its distance, direction, movement, timing and
                        road-network connectivity.
                      </p>
                    </section>

                    <section className="theory-section">
                      <h4>03 / Random Forest Classification</h4>
                      <p>
                        The integrated Random Forest classifier processes 25 features
                        for each candidate. It predicts whether a candidate is a
                        relevant road segment, producing class probabilities that are
                        used by the route-decoding stage.
                      </p>
                    </section>

                    <section className="theory-section">
                      <h4>04 / HMM and Viterbi Decoding</h4>
                      <p>
                        Selecting the highest-scoring candidate independently at every
                        GPS point can produce an inconsistent route. The Hidden Markov
                        Model represents relationships between consecutive candidate
                        states. The Viterbi algorithm finds a high-scoring sequence
                        across the observations using these transition relationships.
                      </p>
                    </section>

                    <section className="theory-section">
                      <h4>05 / Important Limitations</h4>
                      <p>
                        The final route depends on the quality of GPS observations,
                        candidate generation, classifier scores and road-network
                        connectivity. If the correct road is missing from the candidate
                        set, the decoder cannot select it.
                      </p>
                    </section>
                  </div>
                </div>
              )}

              {activeTab === "network" && (
                <div className="info-content">
                  <h2>Road Network</h2>
                  <p>Road graph used by the map-matching and route reconstruction pipeline.</p>
                  <div className="info-stat-grid">
                    <div className="info-stat">
                      <strong>16,284</strong>
                      <span>Road arcs</span>
                    </div>
                    <div className="info-stat">
                      <strong>Directed</strong>
                      <span>Arc connectivity</span>
                    </div>
                    <div className="info-stat">
                      <strong>HMM</strong>
                      <span>Transition model</span>
                    </div>
                  </div>
                  <h3>Network processing</h3>
                  <div className="info-pipeline">
                    <span>Road nodes and arcs</span>
                    <span>↓</span>
                    <span>Directed road graph</span>
                    <span>↓</span>
                    <span>HMM transition probabilities</span>
                    <span>↓</span>
                    <span>Viterbi route decoding</span>
                  </div>
                  <p className="info-note">Arc IDs preserve the original row indices in the network data.</p>
                </div>
              )}

              {activeTab === "analytics" && (
                <AnalyticsDashboard />
              )}
              {activeTab === "data" && (
                <div className="info-content">
                  <h2>Dataset Explorer</h2>
                  <p>Dataset partitions used by the PathSnap project.</p>
                  <div className="info-stat-grid">
                    <div className="info-stat">
                      <strong>80</strong>
                      <span>Training trips</span>
                    </div>
                    <div className="info-stat">
                      <strong>10</strong>
                      <span>Student-test trips</span>
                    </div>
                    <div className="info-stat">
                      <strong>10</strong>
                      <span>Reserved evaluator trips</span>
                    </div>
                  </div>
                  <h3>Data partitions</h3>
                  <div className="info-data-row">
                    <span>Training</span><strong>80 trips</strong>
                  </div>
                  <div className="info-data-row">
                    <span>Student Test</span><strong>10 trips</strong>
                  </div>
                  <div className="info-data-row">
                    <span>Evaluator Test</span><strong>10 reserved trips</strong>
                  </div>
                  <p className="info-note">Evaluator reference labels remain reserved for final server-side evaluation.</p>
                </div>
              )}
            </div>
          ) : (
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
              {prediction?.map_url ? (
                <iframe
                  className="map-frame"
                  title={`PathSnap reconstructed route ${prediction.trip_id}`}
                  src={`${API_URL}${prediction.map_url}`}
                />
              ) : (
                <div className="map-center">
                  <div className="map-crosshair">
                    <Crosshair size={39} />
                  </div>
                  <h3>GEOSPATIAL VIEWPORT</h3>
                  <p>Awaiting GPS observations and road-network geometry</p>
                  <span>SELECT A TRIP TO RECONSTRUCT ITS ROUTE</span>
                </div>
              )}
            </div>

            <div className="map-bottom-overlay">
              <span><i className="legend-dot cyan-dot" /> GPS OBSERVATIONS</span>
              <span><i className="legend-dot purple-dot" /> ROAD NETWORK</span>
              <span><i className="legend-dot red-dot" /> PREDICTED ROUTE</span>
            </div>
            </div>
          )}

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
              {loading ? "PROCESSING..." : "[>]  RECONSTRUCT ROUTE"}
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
              <strong className={prediction ? "green-text" : "muted-text"}>
                {prediction ? "ACTIVE" : "READY"}
              </strong>
            </div>

            <div className="detail-row">
              <span>HMM Decoder</span>
              <strong className={prediction?.status === "hmm_viterbi" ? "green-text" : "muted-text"}>
                {prediction?.status === "hmm_viterbi" ? "ACTIVE" : "READY"}
              </strong>
            </div>
          </section>

          <section className="detail-card">
            <PanelTitle>MODEL PERFORMANCE</PanelTitle>

            <div className="performance-placeholder">
              <div className="performance-ring">
                <span>
                  {typeof accuracy === "number"
                    ? `${(accuracy * 100).toFixed(1)}%`
                    : "N/A"}
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
                {typeof macroF1 === "number" ? macroF1.toFixed(4) : "N/A"}
              </strong>
            </div>

            <div className="detail-row">
              <span>Log Loss</span>
              <strong>
                {typeof logLoss === "number" ? logLoss.toFixed(4) : "N/A"}
              </strong>
            </div>

            <div className="detail-row">
              <span>Dataset</span>
              <strong>{prediction ? "Student Test" : "Awaiting data"}</strong>
            </div>

            <div className="detail-row">
              <span>Evaluation type</span>
              <strong>{prediction ? "Candidate Classification" : "N/A"}</strong>
            </div>

            <div className="detail-row">
              <span>Evaluated trip</span>
              <strong>{prediction?.trip_id || "N/A"}</strong>
            </div>

            <div className="detail-row">
              <span>Evaluation status</span>
              <strong className={prediction?.evaluation ? "green-text" : "muted-text"}>
                {prediction?.evaluation ? "EVALUATED" : "AWAITING"}
              </strong>
            </div>
          </section>

          <section className="detail-card route-performance-card">
            <PanelTitle>ROUTE RECONSTRUCTION PERFORMANCE</PanelTitle>

            <div className="route-metric-grid">
              <div className="route-metric">
                <span>ROUTE ACCURACY</span>
                <strong>
                  {typeof prediction?.route_evaluation?.accuracy === "number"
                    ? `${(prediction.route_evaluation.accuracy * 100).toFixed(2)}%`
                    : "N/A"}
                </strong>
              </div>

              <div className="route-metric">
                <span>ROUTE MACRO F1</span>
                <strong>
                  {typeof prediction?.route_evaluation?.macro_f1 === "number"
                    ? prediction.route_evaluation.macro_f1.toFixed(4)
                    : "N/A"}
                </strong>
              </div>

              <div className="route-metric">
                <span>ROUTE CONTINUITY</span>
                <strong>
                  {typeof prediction?.route_evaluation?.continuity === "number"
                    ? `${(prediction.route_evaluation.continuity * 100).toFixed(2)}%`
                    : "N/A"}
                </strong>
              </div>

              <div className="route-metric">
                <span>CANDIDATE MISS RATE</span>
                <strong>
                  {typeof prediction?.route_evaluation?.candidate_miss_rate === "number"
                    ? `${(prediction.route_evaluation.candidate_miss_rate * 100).toFixed(3)}%`
                    : "N/A"}
                </strong>
              </div>
            </div>

            <div className="detail-row">
              <span>Viterbi latency / observation</span>
              <strong>
                {typeof prediction?.route_evaluation?.decoder_latency_ms_per_observation === "number"
                  ? `${prediction.route_evaluation.decoder_latency_ms_per_observation.toFixed(4)} ms`
                  : "N/A"}
              </strong>
            </div>

            <div className="detail-row">
              <span>Route evaluation</span>
              <strong className={prediction?.route_evaluation ? "green-text" : "muted-text"}>
                {prediction?.route_evaluation ? "COMPLETED" : "AWAITING"}
              </strong>
            </div>

            <p className="route-metric-note">
              Evaluated server-side against the reference route. Reference labels are not exposed to this interface.
            </p>
          </section>

          <section className="detail-card">
            <PanelTitle>PROCESSING DETAILS</PanelTitle>

            <div className="detail-row">
              <span>Selected trip</span>
              <strong>{selectedTrip || "N/A"}</strong>
            </div>

            <div className="detail-row">
              <span>Processing time</span>
              <strong>
                {typeof prediction?.processing_time_seconds === "number"
                  ? `${prediction.processing_time_seconds.toFixed(3)} s`
                  : "N/A"}
              </strong>
            </div>

            <div className="detail-row">
              <span>Predicted segments</span>
              <strong>
                {prediction?.predicted_route?.length ?? "N/A"}
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

