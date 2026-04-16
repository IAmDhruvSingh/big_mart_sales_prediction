import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import datasetCsv from "../Train.csv?raw";

const DATA_SOURCE_URL =
  "https://raw.githubusercontent.com/gokulnpc/BigMart-Sales-Prediction/main/Train.csv";

type ProcessedRow = {
  itemWeight: number;
  itemFatContent: string;
  itemVisibility: number;
  itemType: string;
  itemMRP: number;
  outletIdentifier: string;
  outletEstablishmentYear: number;
  outletSize: string;
  outletLocationType: string;
  outletType: string;
  itemOutletSales: number;
};

type NumericKey = "itemWeight" | "itemVisibility" | "itemMRP" | "outletEstablishmentYear";

type CategoricalKey =
  | "itemFatContent"
  | "itemType"
  | "outletIdentifier"
  | "outletSize"
  | "outletLocationType"
  | "outletType";

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

type Processor = {
  numericStats: Record<NumericKey, { mean: number; std: number }>;
  categoryMaps: Record<CategoricalKey, string[]>;
  featureNames: string[];
};

type ModelArtifacts = {
  processor: Processor;
  weights: number[];
};

type ModelMetrics = {
  trainCount: number;
  validationCount: number;
  rmse: number;
  rSquared: number;
  importantFeatures: Array<{ feature: string; coefficient: number }>;
};

type TrainingState = "idle" | "loading" | "ready" | "error";

const NUMERIC_FEATURES: NumericKey[] = [
  "itemWeight",
  "itemVisibility",
  "itemMRP",
  "outletEstablishmentYear",
];

const CATEGORICAL_FEATURES: CategoricalKey[] = [
  "itemFatContent",
  "itemType",
  "outletIdentifier",
  "outletSize",
  "outletLocationType",
  "outletType",
];

const emptyForm: FormState = {
  itemWeight: "",
  itemFatContent: "",
  itemVisibility: "",
  itemType: "",
  itemMRP: "",
  outletIdentifier: "",
  outletEstablishmentYear: "",
  outletSize: "",
  outletLocationType: "",
  outletType: "",
};

function normalizeFatContent(value: string): string {
  const clean = value.trim().toLowerCase();
  if (clean === "lf" || clean === "low fat") {
    return "Low Fat";
  }
  if (clean === "reg") {
    return "Regular";
  }
  if (clean === "regular") {
    return "Regular";
  }
  return clean.length > 0 ? value.trim() : "Unknown";
}

function parseCsv(csv: string): Array<Record<string, string>> {
  const lines = csv.trim().split(/\r?\n/);
  if (lines.length <= 1) {
    return [];
  }

  const headers = lines[0].split(",");
  const rows: Array<Record<string, string>> = [];

  for (let i = 1; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) {
      continue;
    }
    const values = line.split(",");
    if (values.length !== headers.length) {
      continue;
    }
    const entry: Record<string, string> = {};
    headers.forEach((header, index) => {
      entry[header] = values[index];
    });
    rows.push(entry);
  }

  return rows;
}

function toNumber(value: string): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function preprocessRows(rawRows: Array<Record<string, string>>): ProcessedRow[] {
  const weightValues = rawRows
    .map((row) => toNumber(row.Item_Weight))
    .filter((value) => Number.isFinite(value));

  const meanWeight =
    weightValues.reduce((sum, value) => sum + value, 0) / Math.max(1, weightValues.length);

  return rawRows
    .map((row) => {
      const sales = toNumber(row.Item_Outlet_Sales);
      if (!Number.isFinite(sales)) {
        return null;
      }

      const rawWeight = toNumber(row.Item_Weight);
      const itemWeight = Number.isFinite(rawWeight) ? rawWeight : meanWeight;

      return {
        itemWeight,
        itemFatContent: normalizeFatContent(row.Item_Fat_Content ?? ""),
        itemVisibility: Number.isFinite(toNumber(row.Item_Visibility)) ? toNumber(row.Item_Visibility) : 0,
        itemType: (row.Item_Type ?? "Unknown").trim() || "Unknown",
        itemMRP: Number.isFinite(toNumber(row.Item_MRP)) ? toNumber(row.Item_MRP) : 0,
        outletIdentifier: (row.Outlet_Identifier ?? "Unknown").trim() || "Unknown",
        outletEstablishmentYear: Number.isFinite(toNumber(row.Outlet_Establishment_Year))
          ? toNumber(row.Outlet_Establishment_Year)
          : 2000,
        outletSize: (row.Outlet_Size ?? "").trim() || "Unknown",
        outletLocationType: (row.Outlet_Location_Type ?? "Unknown").trim() || "Unknown",
        outletType: (row.Outlet_Type ?? "Unknown").trim() || "Unknown",
        itemOutletSales: sales,
      };
    })
    .filter((row): row is ProcessedRow => row !== null);
}

function splitTrainValidation(rows: ProcessedRow[], validationRatio = 0.2) {
  const shuffled = [...rows];
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const validationCount = Math.floor(shuffled.length * validationRatio);
  const validationRows = shuffled.slice(0, validationCount);
  const trainRows = shuffled.slice(validationCount);

  return { trainRows, validationRows };
}

function buildProcessor(trainRows: ProcessedRow[]): Processor {
  const numericStats = {} as Record<NumericKey, { mean: number; std: number }>;
  NUMERIC_FEATURES.forEach((feature) => {
    const values = trainRows.map((row) => row[feature]);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
    numericStats[feature] = { mean, std: Math.sqrt(variance) || 1 };
  });

  const categoryMaps = {} as Record<CategoricalKey, string[]>;
  CATEGORICAL_FEATURES.forEach((feature) => {
    categoryMaps[feature] = Array.from(new Set(trainRows.map((row) => row[feature]))).sort();
  });

  const featureNames: string[] = [
    ...NUMERIC_FEATURES.map((feature) => `numeric:${feature}`),
    ...CATEGORICAL_FEATURES.flatMap((feature) =>
      categoryMaps[feature].map((category) => `category:${feature}=${category}`)
    ),
  ];

  return { numericStats, categoryMaps, featureNames };
}

function encodeRow(processor: Processor, row: Omit<ProcessedRow, "itemOutletSales">): number[] {
  const encoded: number[] = [];

  NUMERIC_FEATURES.forEach((feature) => {
    const stats = processor.numericStats[feature];
    encoded.push((row[feature] - stats.mean) / stats.std);
  });

  CATEGORICAL_FEATURES.forEach((feature) => {
    const categories = processor.categoryMaps[feature];
    categories.forEach((category) => {
      encoded.push(row[feature] === category ? 1 : 0);
    });
  });

  return encoded;
}

function transpose(matrix: number[][]): number[][] {
  return matrix[0].map((_, columnIndex) => matrix.map((row) => row[columnIndex]));
}

function multiplyMatrices(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = b[0].length;
  const shared = b.length;
  const result = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let i = 0; i < rows; i += 1) {
    for (let k = 0; k < shared; k += 1) {
      for (let j = 0; j < cols; j += 1) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

function solveLinearSystem(a: number[][], b: number[]): number[] | null {
  const n = a.length;
  const matrix = a.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i += 1) {
    let pivotRow = i;
    for (let r = i + 1; r < n; r += 1) {
      if (Math.abs(matrix[r][i]) > Math.abs(matrix[pivotRow][i])) {
        pivotRow = r;
      }
    }

    if (Math.abs(matrix[pivotRow][i]) < 1e-10) {
      return null;
    }

    if (pivotRow !== i) {
      [matrix[i], matrix[pivotRow]] = [matrix[pivotRow], matrix[i]];
    }

    const pivot = matrix[i][i];
    for (let c = i; c <= n; c += 1) {
      matrix[i][c] /= pivot;
    }

    for (let r = 0; r < n; r += 1) {
      if (r === i) {
        continue;
      }
      const factor = matrix[r][i];
      for (let c = i; c <= n; c += 1) {
        matrix[r][c] -= factor * matrix[i][c];
      }
    }
  }

  return matrix.map((row) => row[n]);
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function calculateRmse(actual: number[], predicted: number[]): number {
  const mse = actual.reduce((sum, value, index) => sum + (value - predicted[index]) ** 2, 0) / actual.length;
  return Math.sqrt(mse);
}

function calculateRSquared(actual: number[], predicted: number[]): number {
  const mean = actual.reduce((sum, value) => sum + value, 0) / actual.length;
  const total = actual.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  const residual = actual.reduce((sum, value, index) => sum + (value - predicted[index]) ** 2, 0);
  return 1 - residual / total;
}

function trainLinearModel(trainRows: ProcessedRow[], validationRows: ProcessedRow[]) {
  const processor = buildProcessor(trainRows);
  const encodedTrain = trainRows.map((row) => [1, ...encodeRow(processor, row)]);
  const yTrain = trainRows.map((row) => row.itemOutletSales);

  const xT = transpose(encodedTrain);
  const xTx = multiplyMatrices(xT, encodedTrain);
  const lambda = 0.8;

  for (let i = 1; i < xTx.length; i += 1) {
    xTx[i][i] += lambda;
  }

  const yColumn = yTrain.map((value) => [value]);
  const xTyColumn = multiplyMatrices(xT, yColumn);
  const xTy = xTyColumn.map((row) => row[0]);

  const weights = solveLinearSystem(xTx, xTy);
  if (!weights) {
    throw new Error("Model training failed because the feature matrix is singular.");
  }

  const model: ModelArtifacts = { processor, weights };
  const validationActual = validationRows.map((row) => row.itemOutletSales);
  const validationPredicted = validationRows.map((row) => {
    const encoded = [1, ...encodeRow(processor, row)];
    return Math.max(0, dot(encoded, weights));
  });

  const metrics: ModelMetrics = {
    trainCount: trainRows.length,
    validationCount: validationRows.length,
    rmse: calculateRmse(validationActual, validationPredicted),
    rSquared: calculateRSquared(validationActual, validationPredicted),
    importantFeatures: processor.featureNames
      .map((feature, index) => ({ feature, coefficient: weights[index + 1] }))
      .sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient))
      .slice(0, 6),
  };

  return { model, metrics };
}

function predictSales(model: ModelArtifacts, form: FormState): number {
  const row = {
    itemWeight: Number(form.itemWeight),
    itemFatContent: form.itemFatContent,
    itemVisibility: Number(form.itemVisibility),
    itemType: form.itemType,
    itemMRP: Number(form.itemMRP),
    outletIdentifier: form.outletIdentifier,
    outletEstablishmentYear: Number(form.outletEstablishmentYear),
    outletSize: form.outletSize,
    outletLocationType: form.outletLocationType,
    outletType: form.outletType,
  };

  const encoded = [1, ...encodeRow(model.processor, row as Omit<ProcessedRow, "itemOutletSales">)];
  return Math.max(0, dot(encoded, model.weights));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export default function App() {
  const predictorRef = useRef<HTMLElement | null>(null);

  const [trainingState, setTrainingState] = useState<TrainingState>("idle");
  const [statusMessage, setStatusMessage] = useState("Waiting to start training...");
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [model, setModel] = useState<ModelArtifacts | null>(null);
  const [form, setForm] = useState<FormState>(emptyForm);
  const [predictedSales, setPredictedSales] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const train = async () => {
      try {
        setTrainingState("loading");
        setErrorMessage(null);
        setStatusMessage("Loading BigMart dataset...");
        setLoadingProgress(15);

        let csv = datasetCsv;
        if (!csv) {
          const response = await fetch(DATA_SOURCE_URL);
          if (!response.ok) {
            throw new Error("Unable to download dataset");
          }
          csv = await response.text();
        }
        await sleep(250);

        setStatusMessage("Parsing and cleaning product-store data...");
        setLoadingProgress(45);
        const rawRows = parseCsv(csv);
        const processedRows = preprocessRows(rawRows);
        if (processedRows.length < 5000) {
          throw new Error("Dataset appears incomplete.");
        }

        await sleep(250);
        setStatusMessage("Training ridge regression model...");
        setLoadingProgress(75);
        const { trainRows, validationRows } = splitTrainValidation(processedRows);
        const trained = trainLinearModel(trainRows, validationRows);

        setModel(trained.model);
        setMetrics(trained.metrics);
        const firstRow = processedRows[0];
        setForm({
          itemWeight: firstRow.itemWeight.toFixed(2),
          itemFatContent: firstRow.itemFatContent,
          itemVisibility: firstRow.itemVisibility.toFixed(4),
          itemType: firstRow.itemType,
          itemMRP: firstRow.itemMRP.toFixed(2),
          outletIdentifier: firstRow.outletIdentifier,
          outletEstablishmentYear: String(firstRow.outletEstablishmentYear),
          outletSize: firstRow.outletSize,
          outletLocationType: firstRow.outletLocationType,
          outletType: firstRow.outletType,
        });

        setStatusMessage("Model ready for prediction.");
        setLoadingProgress(100);
        setTrainingState("ready");
      } catch (error) {
        setTrainingState("error");
        setErrorMessage(error instanceof Error ? error.message : "Unknown error");
        setStatusMessage("Training failed.");
      }
    };

    void train();
  }, []);

  const categoricalOptions = useMemo(() => {
    if (!model) {
      return null;
    }
    return model.processor.categoryMaps;
  }, [model]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!model) {
      return;
    }
    const value = predictSales(model, form);
    setPredictedSales(value);
  };

  const scrollToPredictor = () => {
    predictorRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <main className="bg-slate-950 text-slate-100">
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
            Predict outlet-level product sales using the BigMart Kaggle dataset and an in-browser machine
            learning pipeline.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <button
              type="button"
              onClick={scrollToPredictor}
              className="rounded-md bg-cyan-400 px-6 py-3 font-semibold text-slate-900 transition hover:bg-cyan-300"
            >
              Open Prediction Workspace
            </button>
            <a
              href={DATA_SOURCE_URL}
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
            The model trains automatically on application load with train-validation split. Update product and
            outlet details to estimate expected sales.
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
                  {(categoricalOptions?.itemFatContent ?? []).map((option) => (
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
                  {(categoricalOptions?.itemType ?? []).map((option) => (
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
                  {(categoricalOptions?.outletIdentifier ?? []).map((option) => (
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
                  {(categoricalOptions?.outletSize ?? []).map((option) => (
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
                  {(categoricalOptions?.outletLocationType ?? []).map((option) => (
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
                  {(categoricalOptions?.outletType ?? []).map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <button
              type="submit"
              disabled={trainingState !== "ready"}
              className="mt-6 w-full rounded-md bg-cyan-400 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-600 disabled:text-slate-300"
            >
              Predict Item Outlet Sales
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
              <p className="text-xs tracking-[0.2em] text-cyan-200">MODEL STATUS</p>
              <p className="mt-2 text-sm text-slate-200">{statusMessage}</p>
              {trainingState === "loading" && (
                <div className="mt-4 h-2 w-full rounded-full bg-slate-700">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${loadingProgress}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                    className="h-2 rounded-full bg-cyan-400"
                  />
                </div>
              )}
              {errorMessage && <p className="mt-3 text-sm text-rose-300">Error: {errorMessage}</p>}
            </div>

            <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-6">
              <p className="text-xs tracking-[0.2em] text-cyan-200">PREDICTED SALES</p>
              <motion.p
                key={predictedSales ?? 0}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, ease: "easeOut" }}
                className="mt-3 text-4xl font-semibold text-white"
              >
                {predictedSales !== null ? `INR ${predictedSales.toFixed(2)}` : "INR 0.00"}
              </motion.p>
              <p className="mt-3 text-sm text-slate-300">Estimated revenue for selected item and outlet context.</p>
            </div>

            {metrics && (
              <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-6">
                <p className="text-xs tracking-[0.2em] text-cyan-200">VALIDATION METRICS</p>
                <p className="mt-3 text-sm text-slate-300">
                  RMSE: <span className="font-semibold text-white">{metrics.rmse.toFixed(2)}</span>
                </p>
                <p className="mt-1 text-sm text-slate-300">
                  R2 Score: <span className="font-semibold text-white">{metrics.rSquared.toFixed(3)}</span>
                </p>
                <p className="mt-1 text-sm text-slate-300">
                  Train: <span className="font-semibold text-white">{metrics.trainCount}</span> rows | Validation:{" "}
                  <span className="font-semibold text-white">{metrics.validationCount}</span> rows
                </p>

                <div className="mt-4 space-y-2">
                  {metrics.importantFeatures.map((feature) => (
                    <div key={feature.feature} className="flex items-center justify-between gap-4 text-xs text-slate-300">
                      <span className="truncate">{feature.feature.replace("category:", "").replace("numeric:", "")}</span>
                      <span className="font-semibold text-cyan-200">{feature.coefficient.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </section>
    </main>
  );
}
