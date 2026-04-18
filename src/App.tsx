import { FormEvent, useRef, useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

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
  prediction: (formValues: FormState) => number;
  metrics: { rmse: number; r2: number; trainCount: number; valCount: number };
  featureImportance: { name: string; weight: number }[];
  validationGraphInfo: { name: string; Actual: number; Predicted: number }[];
}

export default function App() {
  const predictorRef = useRef<HTMLElement | null>(null);

  const [form, setForm] = useState<FormState>(emptyForm);
  const [predictedSales, setPredictedSales] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingPhase, setTrainingPhase] = useState("Initializing...");
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [model, setModel] = useState<ModelData | null>(null);

  useEffect(() => {
    const trainModel = async () => {
      setIsTraining(true);
      setTrainingProgress(0);
      try {
        setTrainingPhase("Loading Dataset...");
        setTrainingProgress(15);
        const response = await fetch(new URL("../Train.csv", import.meta.url));
        const text = await response.text();
        const rows = text.trim().split("\n");
        const headers = rows[0].split(",");

        setTrainingPhase("Parsing Data...");
        setTrainingProgress(30);
        const rawData = rows.slice(1).map(row => {
          const values = row.split(",");
          return headers.reduce((obj: Record<string, any>, header, i) => {
            obj[header.trim()] = values[i]?.trim();
            return obj;
          }, {});
        });

        // Filter valid rows
        const validData = rawData.filter(row => parseFloat(row.Item_MRP) > 0 && parseFloat(row.Item_Outlet_Sales) > 0);

        if (validData.length > 0) {
          const first = validData[0];
          setForm({
            itemWeight: first.Item_Weight || "12",
            itemFatContent: first.Item_Fat_Content || "Low Fat",
            itemVisibility: first.Item_Visibility || "0",
            itemType: first.Item_Type || "Baking Goods",
            itemMRP: first.Item_MRP || "100",
            outletIdentifier: first.Outlet_Identifier || "OUT049",
            outletEstablishmentYear: first.Outlet_Establishment_Year || "1999",
            outletSize: first.Outlet_Size || "Medium",
            outletLocationType: first.Outlet_Location_Type || "Tier 1",
            outletType: first.Outlet_Type || "Supermarket Type1",
          });
        }

        setTrainingPhase("Shuffling & Splitting Data (80/20)...");
        setTrainingProgress(50);
        
        // Fisher-Yates shuffle
        const shuffled = [...validData];
        for (let i = shuffled.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        
        const splitIdx = Math.floor(shuffled.length * 0.8);
        const trainData = shuffled.slice(0, splitIdx);
        const valData = shuffled.slice(splitIdx);

        // Feature extraction helper
        const extractFeatures = (row: any) => {
          const itemWeight = parseFloat(row.Item_Weight || row.itemWeight) || 12;
          const itemVisibility = parseFloat(row.Item_Visibility || row.itemVisibility) || 0;
          const itemMRP = parseFloat(row.Item_MRP || row.itemMRP) || 100;
          const year = parseFloat(row.Outlet_Establishment_Year || row.outletEstablishmentYear) || 2000;
          
          const f = [
            1, // bias
            itemWeight,
            itemVisibility,
            itemMRP,
            year,
            itemWeight * itemMRP,
            itemMRP * itemMRP,
            Math.log(itemMRP + 1)
          ];

          // One-hot encode categoricals
          CATEGORIES.itemFatContent.forEach(c => f.push((row.Item_Fat_Content || row.itemFatContent) === c ? 1 : 0));
          CATEGORIES.itemType.forEach(c => f.push((row.Item_Type || row.itemType) === c ? 1 : 0));
          CATEGORIES.outletIdentifier.forEach(c => f.push((row.Outlet_Identifier || row.outletIdentifier) === c ? 1 : 0));
          CATEGORIES.outletSize.forEach(c => f.push((row.Outlet_Size || row.outletSize) === c ? 1 : 0));
          CATEGORIES.outletLocationType.forEach(c => f.push((row.Outlet_Location_Type || row.outletLocationType) === c ? 1 : 0));
          CATEGORIES.outletType.forEach(c => f.push((row.Outlet_Type || row.outletType) === c ? 1 : 0));
          
          return f;
        };

        const featureNames = [
          "Bias", "Weight", "Visibility", "MRP", "Year", "Weight*MRP", "MRP²", "log(MRP)",
          ...CATEGORIES.itemFatContent.map(c => `Fat_${c}`),
          ...CATEGORIES.itemType.map(c => `Type_${c}`),
          ...CATEGORIES.outletIdentifier.map(c => `Out_${c}`),
          ...CATEGORIES.outletSize.map(c => `Size_${c}`),
          ...CATEGORIES.outletLocationType.map(c => `Loc_${c}`),
          ...CATEGORIES.outletType.map(c => `OutType_${c}`)
        ];
        
        const numFeatures = featureNames.length;

        setTrainingPhase("Building Feature Matrix...");
        setTrainingProgress(65);

        const trainX = trainData.map(extractFeatures);
        const trainY = trainData.map(row => parseFloat(row.Item_Outlet_Sales) || 0);

        setTrainingPhase("Training Ridge Regression Model (λ=0.8)...");
        setTrainingProgress(80);

        // Multiple linear regression using normal equation: X^T * X
        const XTX: number[][] = Array(numFeatures).fill(0).map(() => Array(numFeatures).fill(0));
        const XTy: number[] = Array(numFeatures).fill(0);
        
        for (let i = 0; i < trainX.length; i++) {
          for (let j = 0; j < numFeatures; j++) {
            XTy[j] += trainX[i][j] * trainY[i];
            for (let k = 0; k < numFeatures; k++) {
              XTX[j][k] += trainX[i][j] * trainX[i][k];
            }
          }
        }

        // Add Ridge Regularization (lambda = 0.8)
        const lambda = 0.8;
        for (let i = 1; i < numFeatures; i++) {
          XTX[i][i] += lambda; // Skip bias intercept at i=0
        }
        
        const beta = solveLinearSystem(XTX, XTy);

        setTrainingPhase("Evaluating Validation Set...");
        setTrainingProgress(90);

        const prediction = (formStateRow: FormState) => {
          const f = extractFeatures(formStateRow);
          let pred = 0;
          for(let i=0; i<numFeatures; i++) {
            pred += beta[i] * f[i];
          }
          return Math.max(0, pred);
        };

        // Calculate metrics on Validation Set
        const valX = valData.map(extractFeatures);
        const valY = valData.map(row => parseFloat(row.Item_Outlet_Sales) || 0);

        const predictions = valX.map(f => {
          let p = 0;
          for(let i=0; i<numFeatures; i++) p += beta[i] * f[i];
          return Math.max(0, p);
        });

        const meanTarget = valY.reduce((a, b) => a + b, 0) / valY.length;
        const mse = predictions.reduce((sum, p, i) => sum + (p - valY[i]) ** 2, 0) / predictions.length;
        const rmse = Math.sqrt(mse);
        const ssRes = predictions.reduce((sum, p, i) => sum + (p - valY[i]) ** 2, 0);
        const ssTot = valY.reduce((sum, t) => sum + (t - meanTarget) ** 2, 0);
        const r2 = 1 - (ssRes / ssTot);

        // Feature Importance
        const importances = featureNames.slice(1).map((name, idx) => ({
          name,
          weight: beta[idx + 1]
        }));
        importances.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
        const topFeatures = importances.slice(0, 6);

        // Extract validation points for the graph
        const graphDataSubsetLength = Math.min(50, valY.length);
        const validationGraphInfo = [];
        for (let i = 0; i < graphDataSubsetLength; i++) {
          validationGraphInfo.push({
            name: `Sample ${i + 1}`,
            Actual: parseFloat(valY[i].toFixed(2)),
            Predicted: parseFloat(predictions[i].toFixed(2)),
          });
        }

        setTrainingPhase("Complete");
        setTrainingProgress(100);
        setModel({
          prediction,
          metrics: { rmse, r2, trainCount: trainX.length, valCount: valX.length },
          featureImportance: topFeatures,
          validationGraphInfo,
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
      const prediction = model.prediction(form);
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
            <p className="text-white text-xl font-semibold mb-4">{trainingPhase}</p>
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
                  <div className="mt-4 pt-4 border-t border-slate-700/50">
                    <div className="flex justify-between text-xs text-slate-400 mb-1">
                      <span>Training Samples:</span>
                      <span>{model.metrics.trainCount.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>Validation Samples:</span>
                      <span>{model.metrics.valCount.toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {model && (
              <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-6">
                <p className="text-xs tracking-[0.2em] text-cyan-200">TOP 6 FEATURES</p>
                <div className="mt-4 space-y-4">
                  {model.featureImportance.map((feat, idx) => {
                     // Normalize weight for bar width visually
                     const maxWeight = Math.abs(model.featureImportance[0].weight) || 1;
                     const width = Math.max(8, (Math.abs(feat.weight) / maxWeight) * 100);
                     
                     return (
                       <div key={idx} className="space-y-1">
                         <div className="flex justify-between text-sm">
                           <span className="text-slate-300 truncate pr-2">{feat.name}</span>
                           <span className={`shrink-0 ${feat.weight > 0 ? "text-cyan-300" : "text-rose-400"}`}>
                             {feat.weight > 0 ? "+" : ""}{feat.weight.toFixed(2)}
                           </span>
                         </div>
                         <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                           <div 
                             className={`h-full ${feat.weight > 0 ? "bg-cyan-400" : "bg-rose-500"}`} 
                             style={{ width: `${width}%` }} 
                           />
                         </div>
                       </div>
                     );
                  })}
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

        {model && (
          <motion.div
            initial={{ opacity: 0, y: 32 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.1 }}
            transition={{ duration: 0.65, ease: "easeOut", delay: 0.2 }}
            className="mt-10 mx-auto w-full rounded-xl border border-slate-700/80 bg-slate-900/70 p-6 backdrop-blur"
          >
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-white">Validation Accuracy</h3>
              <p className="text-sm text-slate-400">Actual vs Predicted Sales for 50 Random Validation Samples</p>
            </div>
            
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={model.validationGraphInfo}
                  margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis 
                    dataKey="name" 
                    stroke="#94a3b8" 
                    fontSize={12} 
                    tickLine={false} 
                    axisLine={false} 
                    tickFormatter={(value) => value.replace("Sample ", "#")}
                  />
                  <YAxis 
                    stroke="#94a3b8" 
                    fontSize={12} 
                    tickLine={false} 
                    axisLine={false} 
                    tickFormatter={(value) => `₹${value}`}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "12px", boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1)" }}
                    itemStyle={{ color: "#e2e8f0" }}
                    labelStyle={{ color: "#94a3b8", marginBottom: "4px" }}
                  />
                  <Legend wrapperStyle={{ paddingTop: "20px" }} />
                  <Line 
                    type="monotone" 
                    dataKey="Actual" 
                    stroke="#22d3ee" 
                    strokeWidth={3} 
                    dot={false}
                    activeDot={{ r: 6, fill: "#22d3ee", stroke: "#020617", strokeWidth: 2 }} 
                    name="Actual Sales (INR)" 
                  />
                  <Line 
                    type="monotone" 
                    dataKey="Predicted" 
                    stroke="#cbd5e1" 
                    strokeWidth={2} 
                    strokeDasharray="5 5" 
                    dot={false} 
                    activeDot={{ r: 6, fill: "#cbd5e1", stroke: "#020617", strokeWidth: 2 }} 
                    name="Predicted Sales (INR)" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        )}
      </section>
    </main>
  );
}
