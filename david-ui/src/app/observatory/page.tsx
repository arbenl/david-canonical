import { ForecastViewer } from "@/components/forecast-viewer";
import { Nav } from "@/components/nav";

export default function ObservatoryPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Nav />
      <main className="max-w-6xl mx-auto px-4 py-12">
        <div className="mb-12">
          <h1 className="text-3xl font-light tracking-tight text-white mb-4">
            Public Forecast Observatory
          </h1>
          <p className="text-slate-400 max-w-3xl leading-relaxed">
            This observatory provides a read-only view of the latest mathematical simulations of tobacco industry interference.
            Predictions are generated continuously via a joint hierarchical Bayesian MCMC fit across all regional strata.
          </p>
        </div>

        <ForecastViewer />

        {/* Architectural Honesty Banner */}
        <div className="mt-16 p-6 border border-slate-800/50 bg-slate-900/20 rounded-xl">
          <h4 className="text-sm font-bold text-slate-300 mb-2 uppercase tracking-widest">Architectural Honesty</h4>
          <p className="text-xs text-slate-500 leading-relaxed max-w-4xl">
            This interface does not deploy or trigger isolated analytical pipelines. It is a strict read-only viewer over the `forecast_cells` database table. The underlying Dawid-Skene engine and Stan MCMC sampler run in automated batch cycles, evaluating human-adjudicated evidence across all regions simultaneously to preserve hierarchical priors.
          </p>
        </div>
      </main>
    </div>
  );
}
