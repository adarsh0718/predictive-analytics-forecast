// Pure JavaScript Forecasting Engine
// Replaces Pyodide/WebAssembly — runs instantly in-browser with no downloads
// Implements: Linear Regression, Polynomial Regression, Random Forest, confidence intervals

// ─── CSV Parser ────────────────────────────────────────────────────────────────

function parseCSV(text) {
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim().split('\n');
    if (lines.length < 2) return [];

    function splitCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '"' && !inQuotes) { inQuotes = true; continue; }
            if (ch === '"' && inQuotes) { inQuotes = false; continue; }
            if (ch === ',' && !inQuotes) { result.push(current.trim()); current = ''; continue; }
            current += ch;
        }
        result.push(current.trim());
        return result;
    }

    const headers = splitCSVLine(lines[0]);
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const vals = splitCSVLine(lines[i]);
        const row = {};
        headers.forEach((h, idx) => { row[h.trim()] = vals[idx] !== undefined ? vals[idx] : ''; });
        rows.push(row);
    }
    return rows;
}

// ─── Math Helpers ───────────────────────────────────────────────────────────────

function mean(arr) {
    return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function variance(arr, avg) {
    const m = avg !== undefined ? avg : mean(arr);
    return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
}

function stdDev(arr) {
    return Math.sqrt(variance(arr));
}

// ─── Linear Regression (OLS) ────────────────────────────────────────────────────

function linearRegression(X, y) {
    // X is array of feature vectors [[x1,x2,...], ...]
    // For simple 1D: X = [[1], [2], ...]
    const n = X.length;
    const cols = X[0].length;

    // Build X matrix with bias column
    const Xb = X.map(row => [1, ...row]);
    const k = Xb[0].length;

    // Normal equation: β = (XᵀX)⁻¹ Xᵀy — using simple gradient descent for stability
    let weights = new Array(k).fill(0);
    const lr = 1e-10;
    const iters = 2000;
    const yMean = mean(y);
    const yStd = Math.max(stdDev(y), 1);
    const yNorm = y.map(v => (v - yMean) / yStd);

    // Normalize X columns
    const xMeans = Xb[0].map((_, ci) => mean(Xb.map(r => r[ci])));
    const xStds = Xb[0].map((_, ci) => Math.max(stdDev(Xb.map(r => r[ci])), 1));
    // Bias column (index 0) stays as-is
    xMeans[0] = 0; xStds[0] = 1;
    const XbNorm = Xb.map(row => row.map((v, ci) => (v - xMeans[ci]) / xStds[ci]));

    for (let iter = 0; iter < iters; iter++) {
        const preds = XbNorm.map(row => row.reduce((s, v, ci) => s + v * weights[ci], 0));
        const errors = preds.map((p, i) => p - yNorm[i]);
        weights = weights.map((w, ci) => {
            const grad = errors.reduce((s, e, i) => s + e * XbNorm[i][ci], 0) / n;
            return w - lr * 1e10 * grad;
        });
    }

    // Convert weights back to original scale
    const predict = (xRow) => {
        const xbRow = [1, ...xRow];
        const xbNorm = xbRow.map((v, ci) => (v - xMeans[ci]) / xStds[ci]);
        const normPred = xbNorm.reduce((s, v, ci) => s + v * weights[ci], 0);
        return normPred * yStd + yMean;
    };

    return { predict };
}

// ─── Polynomial Feature Builder ─────────────────────────────────────────────────

function polyFeatures(x, degree) {
    // x is a 1D array of values; returns array of feature vectors
    return x.map(v => Array.from({ length: degree }, (_, i) => v ** (i + 1)));
}

// ─── Simple Decision Tree (for Random Forest) ───────────────────────────────────

function buildDecisionTree(X, y, maxDepth, minSamples, depth = 0) {
    const n = X.length;
    const avg = mean(y);

    if (depth >= maxDepth || n <= minSamples) {
        return { isLeaf: true, value: avg };
    }

    let bestGain = Infinity;
    let bestFeature = 0;
    let bestThreshold = 0;

    // Try random subset of features (for forest diversity)
    const featureCount = X[0].length;
    const featuresToTry = Math.max(1, Math.ceil(Math.sqrt(featureCount)));
    const featureIndices = shuffled([...Array(featureCount).keys()]).slice(0, featuresToTry);

    featureIndices.forEach(fi => {
        const vals = X.map(r => r[fi]);
        const thresholds = [...new Set(vals)].sort((a, b) => a - b);
        thresholds.forEach(t => {
            const leftY = y.filter((_, i) => X[i][fi] <= t);
            const rightY = y.filter((_, i) => X[i][fi] > t);
            if (leftY.length === 0 || rightY.length === 0) return;
            const mse = (leftY.length * variance(leftY) + rightY.length * variance(rightY)) / n;
            if (mse < bestGain) {
                bestGain = mse;
                bestFeature = fi;
                bestThreshold = t;
            }
        });
    });

    const leftIdx = X.map((r, i) => r[bestFeature] <= bestThreshold ? i : -1).filter(i => i >= 0);
    const rightIdx = X.map((r, i) => r[bestFeature] > bestThreshold ? i : -1).filter(i => i >= 0);

    if (leftIdx.length === 0 || rightIdx.length === 0) {
        return { isLeaf: true, value: avg };
    }

    return {
        isLeaf: false,
        feature: bestFeature,
        threshold: bestThreshold,
        left: buildDecisionTree(leftIdx.map(i => X[i]), leftIdx.map(i => y[i]), maxDepth, minSamples, depth + 1),
        right: buildDecisionTree(rightIdx.map(i => X[i]), rightIdx.map(i => y[i]), maxDepth, minSamples, depth + 1)
    };
}

function predictTree(tree, x) {
    if (tree.isLeaf) return tree.value;
    return x[tree.feature] <= tree.threshold
        ? predictTree(tree.left, x)
        : predictTree(tree.right, x);
}

function shuffled(arr) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function buildRandomForest(X, y, nTrees = 20, maxDepth = 6, minSamples = 3) {
    const trees = [];
    for (let t = 0; t < nTrees; t++) {
        // Bootstrap sample
        const bootIdx = Array.from({ length: X.length }, () => Math.floor(Math.random() * X.length));
        const bootX = bootIdx.map(i => X[i]);
        const bootY = bootIdx.map(i => y[i]);
        trees.push(buildDecisionTree(bootX, bootY, maxDepth, minSamples));
    }
    return {
        predict: (x) => mean(trees.map(tree => predictTree(tree, x)))
    };
}

// ─── Metrics ────────────────────────────────────────────────────────────────────

function computeMetrics(actual, predicted) {
    const n = actual.length;
    const yMean = mean(actual);

    const mae = actual.reduce((s, v, i) => s + Math.abs(v - predicted[i]), 0) / n;
    const rmse = Math.sqrt(actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0) / n);
    const ssTot = actual.reduce((s, v) => s + (v - yMean) ** 2, 0);
    const ssRes = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
    const r2 = Math.max(-1, 1 - ssRes / Math.max(ssTot, 1e-10));
    const mape = actual.reduce((s, v, i) => s + Math.abs((v - predicted[i]) / Math.max(Math.abs(v), 1)), 0) / n * 100;
    const residuals = actual.map((v, i) => v - predicted[i]);

    return { mae, rmse, r2, mape, residuals };
}

// ─── Date Utilities ─────────────────────────────────────────────────────────────

function addDays(dateStr, n) {
    const d = new Date(dateStr);
    d.setDate(d.getDate() + n);
    return d.toISOString().slice(0, 10);
}

function addWeeks(dateStr, n) { return addDays(dateStr, n * 7); }
function addMonths(dateStr, n) {
    const d = new Date(dateStr);
    d.setMonth(d.getMonth() + n);
    return d.toISOString().slice(0, 10);
}

function nextDate(dateStr, granularity) {
    if (granularity === 'W') return addWeeks(dateStr, 1);
    if (granularity === 'M') return addMonths(dateStr, 1);
    return addDays(dateStr, 1);
}

function dayOfWeek(dateStr) { return new Date(dateStr).getDay(); }
function monthOfYear(dateStr) { return new Date(dateStr).getMonth() + 1; }
function dayOfYear(dateStr) {
    const d = new Date(dateStr);
    const start = new Date(d.getFullYear(), 0, 0);
    return Math.floor((d - start) / 86400000);
}

// ─── Feature Engineering ─────────────────────────────────────────────────────────

function buildFeatures(rows, dateCol, targetCol, granularity) {
    // Sort by date
    const sorted = [...rows].sort((a, b) => new Date(a[dateCol]) - new Date(b[dateCol]));
    const dates = sorted.map(r => r[dateCol]);
    const y = sorted.map(r => parseFloat(r[targetCol]));

    const X = dates.map((d, i) => [
        i,                                  // time index
        dayOfWeek(d),                       // cyclic weekly
        monthOfYear(d),                     // seasonal monthly
        Math.sin(2 * Math.PI * i / 7),      // 7-day sine wave
        Math.cos(2 * Math.PI * i / 7),      // 7-day cosine wave
        Math.sin(2 * Math.PI * i / 30.4),   // monthly sine
        Math.cos(2 * Math.PI * i / 30.4),   // monthly cosine
        Math.sin(2 * Math.PI * i / 365),    // yearly sine
        Math.cos(2 * Math.PI * i / 365),    // yearly cosine
    ]);

    return { X, y, dates, sorted };
}

// ─── Core Forecasting Engine ────────────────────────────────────────────────────

function runForecast(csvText, opts) {
    const { modelType, horizon, splitRatio, granularity } = opts;
    const rows = parseCSV(csvText);
    if (!rows.length) throw new Error('Dataset is empty or could not be parsed.');

    // Auto-detect date and target columns
    const headers = Object.keys(rows[0]);
    const dateKeywords = ['date', 'time', 'timestamp', 'order_date', 'sale_date', 'day'];
    let dateCol = headers.find(h => dateKeywords.some(k => h.toLowerCase().includes(k)));
    if (!dateCol) throw new Error("Dataset must contain a date-like column (e.g. 'Date', 'Timestamp').");

    // Target: first numeric column that isn't the date
    const targetKeywords = ['sales', 'revenue', 'amount', 'value', 'price', 'units', 'count'];
    let targetCol = headers.find(h => h !== dateCol && targetKeywords.some(k => h.toLowerCase().includes(k)));
    if (!targetCol) targetCol = headers.find(h => h !== dateCol && !isNaN(parseFloat(rows[0][h])));
    if (!targetCol) throw new Error("Dataset must contain at least one numeric target column.");

    const { X, y, dates, sorted } = buildFeatures(rows, dateCol, targetCol, granularity);
    const n = X.length;
    const splitAt = Math.floor(n * splitRatio);

    const XTrain = X.slice(0, splitAt);
    const yTrain = y.slice(0, splitAt);
    const XTest = X.slice(splitAt);
    const yTest = y.slice(splitAt);
    const testDates = dates.slice(splitAt);

    // ── Train model ────────────────────────────────────────────────────────────
    let model;
    if (modelType === 'linear') {
        model = linearRegression(XTrain, yTrain);
    } else if (modelType === 'polynomial') {
        // Expand features with polynomial degree-2 interaction on time index
        const polyXTrain = XTrain.map(row => {
            const t = row[0];
            return [...row, t ** 2, t ** 3];
        });
        const polyXTest = XTest.map(row => {
            const t = row[0];
            return [...row, t ** 2, t ** 3];
        });
        const polyModel = linearRegression(polyXTrain, yTrain);
        // Wrap in same interface with expanded features
        model = {
            predict: (xRow) => {
                const t = xRow[0];
                return polyModel.predict([...xRow, t ** 2, t ** 3]);
            }
        };
    } else {
        // Random Forest
        model = buildRandomForest(XTrain, yTrain, 25, 7, 3);
    }

    // ── Compute test predictions and metrics ────────────────────────────────────
    const testPredictions = XTest.map(row => Math.max(0, model.predict(row)));
    const metrics = computeMetrics(yTest, testPredictions);

    // ── Build forecast for future horizon ──────────────────────────────────────
    const lastDate = dates[dates.length - 1];
    const resStd = stdDev(metrics.residuals);
    const forecastData = [];
    let curDate = lastDate;
    for (let h = 1; h <= horizon; h++) {
        curDate = nextDate(curDate, granularity);
        const futureIdx = n + h - 1;
        const xRow = [
            futureIdx,
            dayOfWeek(curDate),
            monthOfYear(curDate),
            Math.sin(2 * Math.PI * futureIdx / 7),
            Math.cos(2 * Math.PI * futureIdx / 7),
            Math.sin(2 * Math.PI * futureIdx / 30.4),
            Math.cos(2 * Math.PI * futureIdx / 30.4),
            Math.sin(2 * Math.PI * futureIdx / 365),
            Math.cos(2 * Math.PI * futureIdx / 365),
        ];
        const pred = Math.max(0, model.predict(xRow));
        const margin = resStd * 1.96 * (1 + h / (horizon * 2)); // widen CI over time
        forecastData.push({
            date: curDate,
            prediction: Math.round(pred),
            lower_bound: Math.round(Math.max(0, pred - margin)),
            upper_bound: Math.round(pred + margin)
        });
    }

    // ── Build historical data for charts ───────────────────────────────────────
    const trainPredictions = XTrain.map(row => Math.max(0, model.predict(row)));
    const historicalData = dates.map((d, i) => ({
        date: d,
        actual: y[i],
        fitted: i < splitAt ? Math.round(trainPredictions[i]) : Math.round(testPredictions[i - splitAt])
    }));

    // ── Test segment chart data ─────────────────────────────────────────────────
    const testData = testDates.map((d, i) => ({
        date: d,
        actual: yTest[i],
        predicted: Math.round(testPredictions[i])
    }));

    return {
        metrics: {
            r2: metrics.r2.toFixed(4),
            mae: metrics.mae.toFixed(2),
            rmse: metrics.rmse.toFixed(2),
            mape: metrics.mape.toFixed(2)
        },
        historical_data: historicalData,
        test_data: testData,
        forecast_data: forecastData,
        residuals: metrics.residuals
    };
}

// ─── Embedded Default Dataset ───────────────────────────────────────────────────
// First 120 rows embedded so the page loads without any fetch (zero wait time)

const DEFAULT_CSV_LINES = [
"Date,Sales_Revenue,Marketing_Spend,Store_Traffic,Discount_Percent,Is_Holiday",
"2023-01-01,148104.91,10703.05,2243,20,1","2023-01-02,50861.16,2599.85,751,5,0",
"2023-01-03,53271.91,2123.51,808,10,0","2023-01-04,61713.51,5704.89,1286,0,0",
"2023-01-05,51127.3,2279.99,891,5,0","2023-01-06,88533.34,7119.57,1410,5,0",
"2023-01-07,111196.22,8514.58,1997,0,0","2023-01-08,123394.57,10911.31,2326,0,0",
"2023-01-09,54129.69,3385.36,1068,5,0","2023-01-10,56712.33,3204.72,1014,0,0",
"2023-01-11,52847.18,2642.54,869,5,0","2023-01-12,52298.13,2382.58,948,0,0",
"2023-01-13,73295.68,5498.12,1472,0,0","2023-01-14,106820.45,9013.17,2105,0,0",
"2023-01-15,132594.77,10926.53,2419,10,0","2023-01-16,55281.36,2883.02,943,5,0",
"2023-01-17,52347.3,2523.15,888,0,0","2023-01-18,55023.47,2855.77,961,0,0",
"2023-01-19,55978.69,3034.52,1033,5,0","2023-01-20,53418.47,2553.52,883,5,0",
"2023-01-21,82285.68,7011.18,1544,0,0","2023-01-22,114219.55,9316.18,2189,0,0",
"2023-01-23,122012.97,10520.17,2276,5,0","2023-01-24,51843.93,2465.73,906,5,0",
"2023-01-25,50399.5,2198.09,855,0,0","2023-01-26,56183.64,2985.0,1005,10,0",
"2023-01-27,59756.37,3649.75,1104,0,0","2023-01-28,53834.8,2782.26,912,5,0",
"2023-01-29,79424.15,6610.53,1520,0,0","2023-01-30,105978.1,8957.37,2072,0,0",
"2023-01-31,119637.83,10127.24,2257,0,0","2023-02-01,56278.39,3129.36,972,5,0",
"2023-02-02,52289.69,2511.62,892,0,0","2023-02-03,52853.75,2673.94,899,5,0",
"2023-02-04,56721.65,3218.63,1020,0,0","2023-02-05,53284.88,2618.57,892,5,0",
"2023-02-06,81263.1,6814.96,1548,0,0","2023-02-07,106541.64,8973.56,2096,0,0",
"2023-02-08,122399.77,10343.4,2296,5,0","2023-02-09,55036.61,2879.72,958,0,0",
"2023-02-10,52073.87,2406.72,890,0,0","2023-02-11,54116.43,2831.08,948,5,0",
"2023-02-12,56082.46,3005.63,1013,0,0","2023-02-13,53117.59,2589.39,920,5,0",
"2023-02-14,96847.63,8617.55,1928,15,1","2023-02-15,118022.38,10117.98,2168,5,0",
"2023-02-16,55523.83,2973.46,941,5,0","2023-02-17,52469.98,2545.06,867,0,0",
"2023-02-18,53889.53,2659.04,928,5,0","2023-02-19,56392.62,3180.25,1019,0,0",
"2023-02-20,53282.88,2627.67,903,5,0","2023-02-21,80025.36,6760.3,1484,0,0",
"2023-02-22,104920.8,8787.31,2058,0,0","2023-02-23,115896.13,9823.05,2192,5,0",
"2023-02-24,53793.93,2762.7,925,0,0","2023-02-25,53137.91,2598.33,895,0,0",
"2023-02-26,55481.28,2956.86,995,5,0","2023-02-27,56684.67,3203.46,1027,0,0",
"2023-02-28,54285.76,2703.75,906,5,0","2023-03-01,84069.7,7017.2,1601,0,0",
"2023-03-02,113294.64,9424.4,2136,0,0","2023-03-03,127682.46,10786.58,2321,5,0",
"2023-03-04,54854.27,2836.61,979,5,0","2023-03-05,51874.6,2400.76,886,0,0",
"2023-03-06,54196.16,2807.77,958,5,0","2023-03-07,58182.33,3473.62,1085,0,0",
"2023-03-08,55148.88,2838.88,953,5,0","2023-03-09,84540.36,7086.35,1582,0,0",
"2023-03-10,116029.42,9706.35,2199,0,0","2023-03-11,132093.81,11028.91,2394,5,0",
"2023-03-12,56124.32,3016.31,979,5,0","2023-03-13,53090.87,2567.16,893,0,0",
"2023-03-14,54855.24,2865.39,961,5,0","2023-03-15,59127.79,3634.76,1098,10,0",
"2023-03-16,55473.95,2924.79,963,5,0","2023-03-17,84785.34,7200.08,1614,0,0",
"2023-03-18,116619.29,9733.15,2180,0,0","2023-03-19,132484.4,11051.14,2415,5,0",
"2023-03-20,55706.09,2951.03,989,5,0","2023-03-21,51860.85,2388.29,891,0,0",
"2023-03-22,55220.77,2947.35,971,5,0","2023-03-23,58491.19,3535.44,1082,0,0",
"2023-03-24,55262.76,2892.48,969,5,0","2023-03-25,87207.34,7340.64,1636,0,0",
"2023-03-26,118462.84,9989.51,2207,0,0","2023-03-27,133490.52,11108.38,2416,5,0",
"2023-03-28,55839.42,3004.72,978,5,0","2023-03-29,52218.09,2493.32,893,0,0",
"2023-03-30,55785.95,2975.73,977,5,0","2023-03-31,57963.24,3413.64,1080,10,0",
"2023-04-01,56021.05,2913.59,974,5,0","2023-04-02,88344.04,7462.73,1640,0,0",
"2023-04-03,120413.62,10105.21,2241,0,0","2023-04-04,140697.28,11862.0,2561,10,1",
"2023-04-05,57327.02,3142.96,1007,5,0","2023-04-06,53434.3,2629.08,916,0,0",
"2023-04-07,55700.37,2946.82,977,5,0","2023-04-08,60248.18,3760.9,1124,0,0",
"2023-04-09,56375.87,2999.42,987,5,0","2023-04-10,87741.46,7410.07,1635,0,0",
"2023-04-11,119428.03,9988.47,2218,0,0","2023-04-12,138625.97,11572.49,2488,5,0",
"2023-04-13,56682.76,3108.9,989,5,0","2023-04-14,52994.01,2579.51,910,0,0",
"2023-04-15,56124.57,2985.43,975,5,0","2023-04-16,59849.53,3722.09,1102,0,0",
"2023-04-17,55808.34,2931.72,969,5,0","2023-04-18,87498.41,7368.59,1628,0,0",
"2023-04-19,117875.0,9862.66,2212,0,0","2023-04-20,136890.97,11432.41,2468,5,0",
"2023-04-21,57209.93,3156.11,1002,5,0","2023-04-22,52889.76,2570.39,912,0,0"
];

const DEFAULT_CSV = DEFAULT_CSV_LINES.join('\n');

// ─── State ──────────────────────────────────────────────────────────────────────

let activeCSVText = DEFAULT_CSV;
let activeDatasetName = 'historical_sales.csv';
let isDefaultDataset = true;

// ─── API Intercept Layer ─────────────────────────────────────────────────────────

const _originalFetch = window.fetch.bind(window);

window.fetch = async function(url, options = {}) {
    const urlStr = typeof url === 'string' ? url : (url && url.url ? url.url : String(url));

    // ── POST /api/forecast ────────────────────────────────────────────────────────
    if (urlStr.includes('/api/forecast')) {
        const body = options.body ? JSON.parse(options.body) : {};
        const result = runForecastAPI(body);
        return new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    // ── POST /api/upload ──────────────────────────────────────────────────────────
    if (urlStr.includes('/api/upload')) {
        const formData = options.body;
        const file = formData.get('file');
        activeCSVText = await file.text();
        activeDatasetName = file.name;
        isDefaultDataset = false;

        const body = {
            model_type: formData.get('model_type') || 'linear',
            horizon: parseInt(formData.get('horizon') || '30'),
            split_ratio: parseFloat(formData.get('split_ratio') || '0.8'),
            granularity: formData.get('granularity') || 'D'
        };
        const result = runForecastAPI(body);
        result.dataset_info = { name: activeDatasetName, is_default: false };
        return new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    // ── POST /api/reset ───────────────────────────────────────────────────────────
    if (urlStr.includes('/api/reset')) {
        // Fetch the full CSV from the server since we only embedded 120 rows
        try {
            const resp = await _originalFetch('data/historical_sales.csv');
            if (resp.ok) activeCSVText = await resp.text();
            else activeCSVText = DEFAULT_CSV;
        } catch(e) { activeCSVText = DEFAULT_CSV; }
        activeDatasetName = 'historical_sales.csv';
        isDefaultDataset = true;
        const result = runForecastAPI({ model_type: 'linear', horizon: 30, split_ratio: 0.8, granularity: 'D' });
        result.dataset_info = { name: 'historical_sales.csv', is_default: true };
        return new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    // ── GET /api/export ───────────────────────────────────────────────────────────
    if (urlStr.includes('/api/export')) {
        const params = new URL(urlStr, window.location.origin).searchParams;
        const body = {
            model_type: params.get('model_type') || 'linear',
            horizon: parseInt(params.get('horizon') || '30'),
            split_ratio: parseFloat(params.get('split_ratio') || '0.8'),
            granularity: params.get('granularity') || 'D'
        };
        const result = runForecastAPI(body);
        const csv = generateForecastCSV(result.forecast_data);
        const blob = new Blob([csv], { type: 'text/csv' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'forecast_predictions.csv';
        link.click();
        return new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    return _originalFetch(url, options);
};

// ─── Run Forecast via active CSV ────────────────────────────────────────────────

function runForecastAPI(body) {
    const { model_type = 'linear', horizon = 30, split_ratio = 0.8, granularity = 'D' } = body;
    const result = runForecast(activeCSVText, {
        modelType: model_type,
        horizon: parseInt(horizon),
        splitRatio: parseFloat(split_ratio),
        granularity
    });
    result.dataset_info = { name: activeDatasetName, is_default: isDefaultDataset };
    return result;
}

// ─── CSV Export ─────────────────────────────────────────────────────────────────

function generateForecastCSV(rows) {
    if (!rows || !rows.length) return '';
    const headers = ['Forecast_Date', 'Point_Forecast_Value', 'Lower_Confidence_Bound', 'Upper_Confidence_Bound'];
    const lines = [headers.join(',')];
    rows.forEach(r => lines.push(`${r.date},${r.prediction},${r.lower_bound},${r.upper_bound}`));
    return lines.join('\n');
}

// ─── Kick off initial load immediately on page ready ─────────────────────────────
// The loading spinner will show/hide in app.js; no Pyodide init needed.
