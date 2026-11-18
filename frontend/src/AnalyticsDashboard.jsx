import { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, LineChart, Line
} from "recharts";

const files = {
  models: "/analytics/models.json",
  cv: "/analytics/cross_validation.json",
  decoders: "/analytics/decoders.json",
  tuning: "/analytics/tuning.json",
  optimization: "/analytics/optimization.json",
  tunedComparison: "/analytics/tuned_comparison.json"
};

const number = (value) => Number(value) || 0;
const pct = (value) => `${(number(value) * 100).toFixed(1)}%`;

export default function AnalyticsDashboard() {
  const [data, setData] = useState({
    models: [], cv: [], decoders: [], tuning: [],
    optimization: [], tunedComparison: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;

    Promise.all(
      Object.entries(files).map(async ([key, url]) => {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Unable to load ${key} results`);
        return [key, await response.json()];
      })
    )
      .then((entries) => {
        if (mounted) setData(Object.fromEntries(entries));
      })
      .catch((err) => {
        if (mounted) setError(err.message);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => { mounted = false; };
  }, []);

  const cvData = useMemo(() => {
    const grouped = {};

    data.cv.forEach((row) => {
      const key = row.model;
      if (!grouped[key]) grouped[key] = { model: key, total: 0, count: 0 };
      grouped[key].total += number(row.f1_macro);
      grouped[key].count += 1;
    });

    return Object.values(grouped).map((row) => ({
      model: row.model,
      macroF1: row.total / row.count
    }));
  }, [data.cv]);

  if (loading) {
    return <div className="analytics-state">Loading experiment results...</div>;
  }

  if (error) {
    return <div className="analytics-state">Analytics error: {error}</div>;
  }

  const bestModel = [...data.models].sort(
    (a, b) => number(b.f1_macro) - number(a.f1_macro)
  )[0];

  const averageHMM = data.decoders.length
    ? data.decoders.reduce((sum, row) =>
        sum + number(row.hmm_viterbi_macro_f1), 0
      ) / data.decoders.length
    : null;

  const averageClassifier = data.decoders.length
    ? data.decoders.reduce((sum, row) =>
        sum + number(row.clf_only_macro_f1), 0
      ) / data.decoders.length
    : null;

  const optimizationChart = data.optimization.map((row) => ({
    trial: `Trial ${row.trial_number}`,
    macroF1: number(row.macro_f1)
  }));

  const tunedComparisonChart = data.tunedComparison.filter(
    (row) => row.model.includes("Tuned") ||
      row.model === "Random Forest" ||
      row.model === "HistGradientBoosting"
  ).map((row) => ({
    model: row.model.replace(" Tuned", ""),
    version: row.model.includes("Tuned") ? "Tuned" : "Untuned",
    macroF1: number(row.f1_macro)
  }));

  const tunedComparisonGrouped = tunedComparisonChart.reduce((acc, row) => {
    let item = acc.find((entry) => entry.model === row.model);
    if (!item) {
      item = { model: row.model };
      acc.push(item);
    }
    item[row.version.toLowerCase()] = row.macroF1;
    return acc;
  }, []);
  const decoderChart = data.decoders.map((row) => ({
    trip: String(row.trip_id).padStart(8, "0"),
    classifier: number(row.clf_only_macro_f1),
    hmm: number(row.hmm_viterbi_macro_f1)
  }));

  return (
    <div className="analytics-dashboard">
      <div className="analytics-heading">
        <div>
          <h2>Experiment Analytics</h2>
          <p>Model performance, cross-validation and route decoder evaluation</p>
        </div>
        <span className="analytics-live">EXPERIMENT DATA</span>
      </div>

      <div className="analytics-summary">
        <div className="analytics-card">
          <span>BEST RECORDED MODEL</span>
          <strong>{bestModel?.model || "N/A"}</strong>
          <small>Highest recorded Macro F1 in comparison file</small>
        </div>
        <div className="analytics-card">
          <span>MODEL MACRO F1</span>
          <strong>{bestModel ? number(bestModel.f1_macro).toFixed(4) : "N/A"}</strong>
          <small>{bestModel?.evaluation_type || "No evaluation data"}</small>
        </div>
        <div className="analytics-card">
          <span>DECODER TRIPS</span>
          <strong>{data.decoders.length}</strong>
          <small>Trips in decoder comparison</small>
        </div>
        <div className="analytics-card">
          <span>AVERAGE DECODER MACRO F1</span>
          <strong>{averageHMM === null ? "N/A" : averageHMM.toFixed(4)}</strong>
          <small>
            Classifier-only: {averageClassifier === null ? "N/A" : averageClassifier.toFixed(4)}
          </small>
        </div>
      </div>

      <div className="analytics-grid">
        <section className="analytics-chart-card">
          <h3>MODEL PERFORMANCE COMPARISON</h3>
          <p>Accuracy and Macro F1 from the model comparison experiment</p>
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={data.models} margin={{ top: 15, right: 10, left: 0, bottom: 55 }}>
              <CartesianGrid stroke="#303746" strokeDasharray="3 3" />
              <XAxis dataKey="model" angle={-22} textAnchor="end" interval={0}
                tick={{ fill: "#aab5c8", fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#aab5c8" }} />
              <Tooltip formatter={(v) => pct(v)} />
              <Legend />
              <Bar dataKey="accuracy" name="Accuracy" fill="#A78BFA" />
              <Bar dataKey="f1_macro" name="Macro F1" fill="#22C5D9" />
            </BarChart>
          </ResponsiveContainer>
          <small className="analytics-note">
            KNN uses sampled observations; the other listed models use full validation.
          </small>
        </section>

        <section className="analytics-chart-card">
          <h3>FIVE-FOLD CROSS-VALIDATION</h3>
          <p>Mean Macro F1 across recorded folds for each model</p>
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={cvData} margin={{ top: 15, right: 10, left: 0, bottom: 55 }}>
              <CartesianGrid stroke="#303746" strokeDasharray="3 3" />
              <XAxis dataKey="model" angle={-22} textAnchor="end" interval={0}
                tick={{ fill: "#aab5c8", fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#aab5c8" }} />
              <Tooltip formatter={(v) => number(v).toFixed(4)} />
              <Bar dataKey="macroF1" name="Mean Macro F1" fill="#34D399" />
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="analytics-chart-card">
          <h3>CLASSIFIER VS HMM / VITERBI</h3>
          <p>Macro F1 comparison for each student-test trip</p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={decoderChart} margin={{ top: 15, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid stroke="#303746" strokeDasharray="3 3" />
              <XAxis dataKey="trip" tick={{ fill: "#aab5c8", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#aab5c8" }} />
              <Tooltip formatter={(v) => number(v).toFixed(4)} />
              <Legend />
              <Bar dataKey="classifier" name="Classifier only" fill="#A78BFA" />
              <Bar dataKey="hmm" name="HMM / Viterbi" fill="#22C5D9" />
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="analytics-chart-card">
          <h3>DECODER ACCURACY BY TRIP</h3>
          <p>Classifier-only versus HMM/Viterbi accuracy</p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data.decoders.map((row) => ({
              trip: String(row.trip_id).padStart(8, "0"),
              classifier: number(row.clf_only_accuracy),
              hmm: number(row.hmm_viterbi_accuracy)
            }))}>
              <CartesianGrid stroke="#303746" strokeDasharray="3 3" />
              <XAxis dataKey="trip" tick={{ fill: "#aab5c8", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#aab5c8" }} />
              <Tooltip formatter={(v) => pct(v)} />
              <Legend />
              <Line type="monotone" dataKey="classifier" name="Classifier only"
                stroke="#A78BFA" strokeWidth={2} />
              <Line type="monotone" dataKey="hmm" name="HMM / Viterbi"
                stroke="#22C5D9" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </section>
      </div>

      <section className="analytics-chart-card">
        <h3>BAYESIAN OPTIMISATION — RANDOM FOREST</h3>
        <p>Recorded Macro F1 across optimisation trials</p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={optimizationChart} margin={{ top: 15, right: 20, left: 5, bottom: 10 }}>
            <CartesianGrid stroke="#303746" strokeDasharray="3 3" />
            <XAxis dataKey="trial" tick={{ fill: "#aab5c8", fontSize: 10 }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#aab5c8" }} />
            <Tooltip formatter={(v) => number(v).toFixed(4)} />
            <Line type="monotone" dataKey="macroF1" name="Macro F1"
              stroke="#F59E0B" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
        <small className="analytics-note">
          Each point represents a recorded Random Forest optimisation trial.
        </small>
      </section>

      <section className="analytics-chart-card">
        <h3>TUNED VS UNTUNED MODELS</h3>
        <p>Macro F1 comparison before and after hyperparameter tuning</p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={tunedComparisonGrouped} margin={{ top: 15, right: 10, left: 0, bottom: 35 }}>
            <CartesianGrid stroke="#303746" strokeDasharray="3 3" />
            <XAxis dataKey="model" tick={{ fill: "#aab5c8", fontSize: 10 }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#aab5c8" }} />
            <Tooltip formatter={(v) => number(v).toFixed(4)} />
            <Legend />
            <Bar dataKey="untuned" name="Untuned Macro F1" fill="#A78BFA" />
            <Bar dataKey="tuned" name="Tuned Macro F1" fill="#34D399" />
          </BarChart>
        </ResponsiveContainer>
        <small className="analytics-note">
          Only Random Forest and HistGradientBoosting have tuned and untuned pairs.
        </small>
      </section>
      <section className="analytics-chart-card analytics-table-card">
        <h3>MODEL METRICS TABLE</h3>
        <div className="analytics-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Evaluation</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>Log Loss</th>
                <th>Observation Top-1</th>
              </tr>
            </thead>
            <tbody>
              {data.models.map((row) => (
                <tr key={row.model}>
                  <td>{row.model}</td>
                  <td>{row.evaluation_type}</td>
                  <td>{pct(row.accuracy)}</td>
                  <td>{number(row.f1_macro).toFixed(4)}</td>
                  <td>{number(row.log_loss).toFixed(4)}</td>
                  <td>{pct(row.observation_top1_accuracy)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <small className="analytics-note">
          Results reflect the experiment CSVs currently available in the repository.
          They are not live metrics for the currently selected trip.
        </small>
      </section>
    </div>
  );
}


