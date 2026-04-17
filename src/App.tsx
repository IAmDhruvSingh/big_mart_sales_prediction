import { FormEvent, useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";

type FormState = {
  itemWeight: string;
  itemFatContent: string;
  itemVisibility: string;
  itemType: string;
  itemMRP: string;
  outletIdentifier: string;
  outletEstablishmentYear: string;
  outletSize: string;
  outletLocationType: string;
  outletType: string;
};

const emptyForm: FormState = {
  itemWeight: "",
  itemFatContent: "Low Fat",
  itemVisibility: "",
  itemType: "Baking Goods",
  itemMRP: "",
  outletIdentifier: "OUT049",
  outletEstablishmentYear: "1999",
  outletSize: "Medium",
  outletLocationType: "Tier 1",
  outletType: "Supermarket Type1",
};

const CATEGORIES = {
  itemFatContent: ["Low Fat", "Regular", "Unknown"],
  itemType: [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy",
    "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
    "Health and Hygiene", "Household", "Meat", "Others",
    "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods", "Unknown"
  ],
  outletIdentifier: [
    "OUT010", "OUT013", "OUT017", "OUT018", "OUT019",
    "OUT027", "OUT035", "OUT045", "OUT046", "OUT049", "Unknown"
  ],
  outletSize: ["High", "Medium", "Small", "Unknown"],
  outletLocationType: ["Tier 1", "Tier 2", "Tier 3", "Unknown"],
  outletType: [
    "Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Unknown"
  ],
};

// Gaussian elimination to solve Ax = b
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const matrix = A.map((row, i) => [...row, b[i]]);

  // Forward elimination
  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(matrix[k][i]) > Math.abs(matrix[maxRow][i])) {
        maxRow = k;
      }
    }
    [matrix[i], matrix[maxRow]] = [matrix[maxRow], matrix[i]];

    for (let k = i + 1; k < n; k++) {
      const factor = matrix[k][i] / matrix[i][i];
      for (let j = i; j <= n; j++) {
        matrix[k][j] -= factor * matrix[i][j];
      }
    }
  }

  // Back substitution
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = matrix[i][n];
    for (let j = i + 1; j < n; j++) {
      x[i] -= matrix[i][j] * x[j];
    }
    x[i] /= matrix[i][i];
  }

  return x;
}

interface ModelData {
  prediction: (features: Record<string, number>) => number;
  metrics: { rmse: number; r2: number };
  featureImportance: Record<string, number>;
}

export default function App() {
  const predictorRef = useRef<HTMLElement | null>(null);

  const [form, setForm] = useState<FormState>(emptyForm);
  const [predictedSales, setPredictedSales] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [model, setModel] = useState<ModelData | null>(null);

  useEffect(() => {
    const trainModel = async () => {
      setIsTraining(true);
      setTrainingProgress(0);
      try {
        setTrainingProgress(15);
        const response = await fetch(new URL("../Train.csv", import.meta.url));
        const text = await response.text();
        const rows = text.trim().split("\n");
        const headers = rows[0].split(",");

        setTrainingProgress(30);
        const data = rows.slice(1).map(row => {
          const values = row.split(",");
          return headers.reduce((obj: Record<string, any>, header, i) => {
            obj[header.trim()] = values[i]?.trim();
            return obj;
          }, {});
        });

        setTrainingProgress(50);
        // Multiple linear regression model with feature engineering
        const features: Record<string, number>[] = [];
        const targets: number[] = [];

        data.forEach((row: any) => {
          const itemWeight = parseFloat(row.Item_Weight) || 12;
          const itemVisibility = parseFloat(row.Item_Visibility) || 0;
          const itemMRP = parseFloat(row.Item_MRP) || 100;
          const year = parseFloat(row.Outlet_Establishment_Year) || 2000;
          const sales = parseFloat(row.Item_Outlet_Sales) || 0;

          if (itemMRP > 0 && sales > 0) {
            // Feature engineering: add interaction terms and polynomial features
            features.push({
              weight: itemWeight,
              visibility: itemVisibility,
              mrp: itemMRP,
              year: year,
              weightMrp: itemWeight * itemMRP,
              mrpSquared: itemMRP * itemMRP,
              logMrp: Math.log(itemMRP + 1),
            });
            targets.push(sales);
          }
        });

        setTrainingProgress(75);
        // Multiple linear regression using normal equation
        const n = features.length;
        
        // Create design matrix X (n x 8 for intercept + 7 features)
        const X: number[][] = features.map(f => [
          1, f.weight, f.visibility, f.mrp, f.year, f.weightMrp, f.mrpSquared, f.logMrp
        ]);
        
        // Create target vector y
        const y = targets;
        
        // Calculate X^T * X
        const XTX: number[][] = Array(8).fill(0).map(() => Array(8).fill(0));
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < 8; j++) {
            for (let k = 0; k < 8; k++) {
              XTX[j][k] += X[i][j] * X[i][k];
            }
          }
        }
        
        // Calculate X^T * y
        const XTy: number[] = Array(8).fill(0);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < 8; j++) {
            XTy[j] += X[i][j] * y[i];
          }
        }
        
        // Simple Gaussian elimination to solve XTX * beta = XTy
        const beta = solveLinearSystem(XTX, XTy);

        setTrainingProgress(90);
        const prediction = (features: Record<string, number>) => {
          const pred = 
            beta[0] +
            beta[1] * features.weight +
            beta[2] * features.visibility +
            beta[3] * features.mrp +
            beta[4] * features.year +
            beta[5] * (features.weight * features.mrp) +
            beta[6] * (features.mrp * features.mrp) +
            beta[7] * Math.log(features.mrp + 1);
          return Math.max(0, pred);
        };

        // Calculate metrics
        const predictions = features.map(f => prediction(f));
        const meanTarget = targets.reduce((a, b) => a + b, 0) / targets.length;
        const mse = predictions.reduce((sum, p, i) => sum + (p - targets[i]) ** 2, 0) / predictions.length;
        const rmse = Math.sqrt(mse);
        const ssRes = predictions.reduce((sum, p, i) => sum + (p - targets[i]) ** 2, 0);
        const ssTot = targets.reduce((sum, t) => sum + (t - meanTarget) ** 2, 0);
        const r2 = 1 - (ssRes / ssTot);

        setTrainingProgress(100);
        setModel({
          prediction,
          metrics: { rmse, r2 },
          featureImportance: { weight: 0.6, mrp: 0.3, visibility: 0.1 },
        });
      } catch (error) {
        console.error("Training error:", error);
        setErrorMessage("Failed to train model");
      } finally {
        setIsTraining(false);
      }
    };

    trainModel();
  }, []);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!model) {
      setErrorMessage("Model not ready yet");
      return;
    }

    setIsPredicting(true);
    setErrorMessage(null);

    try {
      const features = {
        weight: parseFloat(form.itemWeight) || 12,
        visibility: parseFloat(form.itemVisibility) || 0,
        mrp: parseFloat(form.itemMRP) || 100,
        year: parseFloat(form.outletEstablishmentYear) || 2000,
      };

      const prediction = model.prediction(features);
      setPredictedSales(prediction);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Prediction failed");
    } finally {
      setIsPredicting(false);
    }
  };

  const scrollToPredictor = () => {
    predictorRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <main className="bg-slate-950 text-slate-100 min-h-screen">
      {isTraining && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 flex items-center justify-center bg-black/50 z-50 backdrop-blur"
        >
          <div className="bg-slate-900 rounded-xl p-8 max-w-md w-full mx-4">
            <p className="text-cyan-200 text-sm tracking-[0.2em] mb-4">MODEL TRAINING</p>
            <p className="text-white text-xl font-semibold mb-4">Preparing ML Model...</p>
            <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <motion.div
                className="h-full bg-linear-to-r from-cyan-400 to-cyan-600"
                initial={{ width: 0 }}
                animate={{ width: `${trainingProgress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <p className="text-slate-400 text-sm mt-3">{trainingProgress}% Complete</p>
          </div>
        </motion.div>
      )}

      <section className="relative flex min-h-screen items-center overflow-hidden">
        <motion.div
          initial={{ scale: 1.08 }}
          animate={{ scale: 1 }}
          transition={{ duration: 1.8, ease: "easeOut" }}
          className="absolute inset-0"
          style={{
            backgroundImage:
              "linear-gradient(rgba(2,6,23,0.68), rgba(2,6,23,0.8)), url('https://images.unsplash.com/photo-1604719312566-8912e9227c6a?auto=format&fit=crop&w=1800&q=80')",
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        />

        <motion.div
          initial={{ opacity: 0, y: 32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: "easeOut", delay: 0.15 }}
          className="relative mx-auto w-full max-w-6xl px-6 py-24 md:px-10"
        >
          <p className="text-sm tracking-[0.28em] text-cyan-200">BIGMART FORECAST LAB</p>
          <h1 className="mt-3 text-5xl font-semibold tracking-tight text-white md:text-7xl">BigMart Vision</h1>
          <p className="mt-6 max-w-2xl text-lg text-slate-200 md:text-xl">
            Predict outlet-level product sales using the BigMart Kaggle dataset with in-browser machine learning.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <button
              type="button"
              onClick={scrollToPredictor}
              disabled={isTraining}
              className="rounded-md bg-cyan-400 px-6 py-3 font-semibold text-slate-900 transition hover:bg-cyan-300 disabled:opacity-50"
            >
              Open Prediction Workspace
            </button>
            <a
              href="https://raw.githubusercontent.com/gokulnpc/BigMart-Sales-Prediction/main/Train.csv"
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-slate-300/40 px-6 py-3 font-semibold text-slate-100 transition hover:border-cyan-300 hover:text-cyan-200"
            >
              View Kaggle Data Source
            </a>
          </div>
        </motion.div>
      </section>

      <section ref={predictorRef} className="mx-auto w-full max-w-6xl px-6 py-20 md:px-10">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <h2 className="text-3xl font-semibold text-white md:text-4xl">Sales Prediction Workspace</h2>
          <p className="mt-3 max-w-3xl text-slate-300">
            Submit item and outlet details to get estimated sales predictions using the trained ML model.
          </p>
        </motion.div>

        <div className="mt-10 grid gap-10 lg:grid-cols-[1.35fr_1fr]">
          <motion.form
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.65, ease: "easeOut" }}
            onSubmit={handleSubmit}
            className="rounded-xl border border-slate-700/80 bg-slate-900/80 p-6 backdrop-blur"
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="space-y-2 text-sm text-slate-200">
                <span>Item Weight</span>
                <input
                  type="number"
                  step="0.01"
                  required
                  value={form.itemWeight}
                  onChange={(event) => setForm((prev) => ({ ...prev, itemWeight: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                  placeholder="e.g. 9.30"
                />
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Item Visibility</span>
                <input
                  type="number"
                  step="0.0001"
                  required
                  value={form.itemVisibility}
                  onChange={(event) => setForm((prev) => ({ ...prev, itemVisibility: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                  placeholder="e.g. 0.016"
                />
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Item MRP</span>
                <input
                  type="number"
                  step="0.01"
                  required
                  value={form.itemMRP}
                  onChange={(event) => setForm((prev) => ({ ...prev, itemMRP: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                  placeholder="e.g. 249.80"
                />
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Outlet Establishment Year</span>
                <input
                  type="number"
                  required
                  value={form.outletEstablishmentYear}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, outletEstablishmentYear: event.target.value }))
                  }
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                  placeholder="e.g. 1999"
                />
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Item Fat Content</span>
                <select
                  required
                  value={form.itemFatContent}
                  onChange={(event) => setForm((prev) => ({ ...prev, itemFatContent: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                >
                  {CATEGORIES.itemFatContent.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Item Type</span>
                <select
                  required
                  value={form.itemType}
                  onChange={(event) => setForm((prev) => ({ ...prev, itemType: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                >
                  {CATEGORIES.itemType.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Outlet Identifier</span>
                <select
                  required
                  value={form.outletIdentifier}
                  onChange={(event) => setForm((prev) => ({ ...prev, outletIdentifier: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                >
                  {CATEGORIES.outletIdentifier.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Outlet Size</span>
                <select
                  required
                  value={form.outletSize}
                  onChange={(event) => setForm((prev) => ({ ...prev, outletSize: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                >
                  {CATEGORIES.outletSize.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Outlet Location Type</span>
                <select
                  required
                  value={form.outletLocationType}
                  onChange={(event) => setForm((prev) => ({ ...prev, outletLocationType: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                >
                  {CATEGORIES.outletLocationType.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-2 text-sm text-slate-200">
                <span>Outlet Type</span>
                <select
                  required
                  value={form.outletType}
                  onChange={(event) => setForm((prev) => ({ ...prev, outletType: event.target.value }))}
                  className="w-full rounded-md border border-slate-600 bg-slate-950 px-3 py-2 text-slate-100 outline-none ring-cyan-300 focus:ring"
                >
                  {CATEGORIES.outletType.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <button
              type="submit"
              disabled={isPredicting || isTraining || !model}
              className="mt-6 w-full rounded-lg bg-linear-to-r from-cyan-400 to-cyan-600 px-6 py-3 font-semibold text-slate-900 transition hover:from-cyan-300 hover:to-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isPredicting ? "Predicting..." : "Predict Item Outlet Sales"}
            </button>
          </motion.form>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.65, ease: "easeOut", delay: 0.1 }}
            className="space-y-6"
          >
            <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-6">
              <p className="text-xs tracking-[0.2em] text-cyan-200">PREDICTED SALES</p>
              <motion.p
                key={predictedSales ?? 0}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, ease: "easeOut" }}
                className="mt-3 text-4xl font-semibold text-white"
              >
                {predictedSales !== null ? `INR ${predictedSales.toFixed(2)}` : "---"}
              </motion.p>
              <p className="mt-3 text-sm text-slate-300">Estimated revenue using Linear Regression.</p>
            </div>

            {model && (
              <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-6">
                <p className="text-xs tracking-[0.2em] text-cyan-200">MODEL METRICS</p>
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between">
                    <span className="text-slate-300">R² Score:</span>
                    <span className="font-semibold text-cyan-300">{model.metrics.r2.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">RMSE:</span>
                    <span className="font-semibold text-cyan-300">{model.metrics.rmse.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            )}

            {errorMessage && (
              <div className="rounded-xl border border-rose-700/80 bg-rose-900/20 p-6">
                <p className="text-xs tracking-[0.2em] text-rose-300">ERROR</p>
                <p className="mt-2 text-sm text-rose-200">{errorMessage}</p>
              </div>
            )}
          </motion.div>
        </div>
      </section>
    </main>
  );
}
