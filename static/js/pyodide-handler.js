// Pyodide WebAssembly Handler for Predictive-Analytics-Forecast
// Intercepts Flask API calls and routes them to an in-browser Python runtime

const originalFetch = window.fetch;
let pyodideInstance = null;
let pyodideReadyPromise = null;

// Initialize Pyodide runtime and load dependencies
async function initPyodide() {
    console.log("[Pyodide] Initializing WebAssembly Python environment...");
    
    // Load Pyodide
    let pyodide = await loadPyodide();
    
    // Load Pandas, Numpy, Scikit-learn, and Micropip
    console.log("[Pyodide] Loading Python packages (pandas, numpy, scikit-learn)...");
    await pyodide.loadPackage(['pandas', 'numpy', 'scikit-learn', 'micropip']);
    
    // Setup Virtual File System structure
    pyodide.FS.mkdir('data');
    
    // Download and write the forecasting engine
    console.log("[Pyodide] Fetching forecasting_engine.py...");
    let engineRes = await originalFetch('forecasting_engine.py');
    let engineCode = await engineRes.text();
    pyodide.FS.writeFile('forecasting_engine.py', engineCode);
    
    // Download and write the default historical dataset
    console.log("[Pyodide] Fetching default historical_sales.csv...");
    let salesRes = await originalFetch('data/historical_sales.csv');
    let salesData = await salesRes.text();
    pyodide.FS.writeFile('data/historical_sales.csv', salesData);
    
    console.log("[Pyodide] Runtime environment is fully ready.");
    pyodideInstance = pyodide;
    return pyodide;
}

// Start loading Pyodide immediately
pyodideReadyPromise = initPyodide();

// Intercept window.fetch to mock backend API endpoints
window.fetch = async function(url, options) {
    const urlStr = typeof url === 'string' ? url : url.url;
    
    // Match API routes
    if (urlStr.includes('/api/forecast')) {
        console.log("[Pyodide Intercept] POST /api/forecast");
        const pyodide = await pyodideReadyPromise;
        
        let body = {};
        if (options && options.body) {
            body = JSON.parse(options.body);
        } else {
            // Extract from URL query params if GET
            const parsedUrl = new URL(urlStr, window.location.origin);
            body = {
                model_type: parsedUrl.searchParams.get('model_type'),
                horizon: parsedUrl.searchParams.get('horizon'),
                split_ratio: parsedUrl.searchParams.get('split_ratio'),
                granularity: parsedUrl.searchParams.get('granularity')
            };
        }
        
        const result = await runForecastInPyodide(
            pyodide,
            body.model_type || 'linear',
            body.horizon !== undefined ? parseInt(body.horizon) : 30,
            body.split_ratio !== undefined ? parseFloat(body.split_ratio) : 0.8,
            body.granularity || 'D'
        );
        
        return new Response(JSON.stringify(result), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
        
    } else if (urlStr.includes('/api/upload')) {
        console.log("[Pyodide Intercept] POST /api/upload");
        const pyodide = await pyodideReadyPromise;
        
        const formData = options.body; // FormData object
        const file = formData.get('file');
        const fileContent = await file.text();
        const filename = file.name;
        
        // Basic validation: must have Date column and a numeric column
        // We will do validation inside Python
        pyodide.FS.writeFile('data/uploaded_dataset.csv', fileContent);
        pyodide.FS.writeFile('data/active_dataset_path.txt', 'data/uploaded_dataset.csv');
        
        const modelType = formData.get('model_type') || 'linear';
        const horizon = parseInt(formData.get('horizon') || '30');
        const splitRatio = parseFloat(formData.get('split_ratio') || '0.8');
        const granularity = formData.get('granularity') || 'D';
        
        const result = await runForecastInPyodide(
            pyodide,
            modelType,
            horizon,
            splitRatio,
            granularity
        );
        
        result.dataset_info = {
            name: filename,
            is_default: false
        };
        
        return new Response(JSON.stringify(result), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
        
    } else if (urlStr.includes('/api/reset')) {
        console.log("[Pyodide Intercept] POST /api/reset");
        const pyodide = await pyodideReadyPromise;
        
        try {
            if (pyodide.FS.analyzePath('data/active_dataset_path.txt').exists) {
                pyodide.FS.unlink('data/active_dataset_path.txt');
            }
        } catch(e) {}
        
        const result = await runForecastInPyodide(pyodide, 'linear', 30, 0.8, 'D');
        result.dataset_info = {
            name: 'historical_sales.csv',
            is_default: true
        };
        
        return new Response(JSON.stringify(result), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
        
    } else if (urlStr.includes('/api/export')) {
        console.log("[Pyodide Intercept] GET /api/export");
        const pyodide = await pyodideReadyPromise;
        
        const parsedUrl = new URL(urlStr, window.location.origin);
        const modelType = parsedUrl.searchParams.get('model_type') || 'linear';
        const horizon = parseInt(parsedUrl.searchParams.get('horizon') || '30');
        const splitRatio = parseFloat(parsedUrl.searchParams.get('split_ratio') || '0.8');
        const granularity = parsedUrl.searchParams.get('granularity') || 'D';
        
        const csvContent = await generateExportCsvInPyodide(
            pyodide,
            modelType,
            horizon,
            splitRatio,
            granularity
        );
        
        return new Response(csvContent, {
            status: 200,
            headers: {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename="forecast_predictions.csv"'
            }
        });
    }
    
    // For anything else, fall back to the original fetch
    return originalFetch(url, options);
};

// Runner helper for forecast
async function runForecastInPyodide(pyodide, modelType, horizon, splitRatio, granularity) {
    const pyCode = `
import json
import os
import pandas as pd
from forecasting_engine import preprocess_and_forecast

def get_active_filepath():
    if os.path.exists('data/active_dataset_path.txt'):
        with open('data/active_dataset_path.txt', 'r') as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return 'data/historical_sales.csv'

# Basic validation wrapper for custom datasets
filepath = get_active_filepath()
if filepath == 'data/uploaded_dataset.csv':
    df = pd.read_csv(filepath)
    has_date = False
    for col in df.columns:
        if col.lower() in ['date', 'time', 'timestamp', 'order_date', 'sale_date']:
            df.rename(columns={col: 'Date'}, inplace=True)
            has_date = True
            break
            
    if not has_date:
        raise Exception("Dataset must contain a date-like column (e.g. 'Date', 'Time').")
        
    # Find numeric target column
    target_col = None
    for col in df.columns:
        if col != 'Date' and pd.api.types.is_numeric_dtype(df[col]):
            df.rename(columns={col: 'Sales'}, inplace=True)
            target_col = 'Sales'
            break
            
    if not target_col:
        raise Exception("Dataset must contain at least one numeric target column.")
        
    df.to_csv(filepath, index=False)

res = preprocess_and_forecast(
    filepath=filepath,
    model_type='${modelType}',
    horizon=${horizon},
    split_ratio=${splitRatio},
    granularity='${granularity}'
)
# Serialize results
json.dumps(res)
`;
    const resultStr = await pyodide.runPythonAsync(pyCode);
    return JSON.parse(resultStr);
}

// Runner helper for export
async function generateExportCsvInPyodide(pyodide, modelType, horizon, splitRatio, granularity) {
    const pyCode = `
import os
import pandas as pd
from forecasting_engine import preprocess_and_forecast

def get_active_filepath():
    if os.path.exists('data/active_dataset_path.txt'):
        with open('data/active_dataset_path.txt', 'r') as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return 'data/historical_sales.csv'

filepath = get_active_filepath()
res = preprocess_and_forecast(
    filepath=filepath,
    model_type='${modelType}',
    horizon=${horizon},
    split_ratio=${splitRatio},
    granularity='${granularity}'
)

forecast_df = pd.DataFrame(res['forecast_data'])
forecast_df.rename(columns={
    "date": "Forecast_Date",
    "prediction": "Point_Forecast_Value",
    "lower_bound": "Lower_Confidence_Bound",
    "upper_bound": "Upper_Confidence_Bound"
}, inplace=True)

csv_str = forecast_df.to_csv(index=False)
csv_str
`;
    const csvContent = await pyodide.runPythonAsync(pyCode);
    return csvContent;
}
