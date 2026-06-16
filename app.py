import os
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
from forecasting_engine import preprocess_and_forecast

app = Flask(__name__, template_folder='templates', static_folder='static')

# Folder configurations
UPLOAD_FOLDER = 'data'
DEFAULT_DATA_PATH = os.path.join('data', 'historical_sales.csv')
ACTIVE_DATA_PATH_FILE = os.path.join('data', 'active_dataset_path.txt')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper to get active dataset filepath
def get_active_filepath():
    if os.path.exists(ACTIVE_DATA_PATH_FILE):
        with open(ACTIVE_DATA_PATH_FILE, 'r') as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return DEFAULT_DATA_PATH

# Helper to set active dataset filepath
def set_active_filepath(path):
    with open(ACTIVE_DATA_PATH_FILE, 'w') as f:
        f.write(path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forecast', methods=['GET', 'POST'])
def run_forecast():
    try:
        # Get parameters from query string or JSON payload
        if request.method == 'POST':
            params = request.get_json() or {}
        else:
            params = request.args or {}
            
        model_type = params.get('model_type', 'linear')
        horizon = int(params.get('horizon', 30))
        split_ratio = float(params.get('split_ratio', 0.8))
        granularity = params.get('granularity', 'D')
        
        filepath = get_active_filepath()
        
        results = preprocess_and_forecast(
            filepath=filepath,
            model_type=model_type,
            horizon=horizon,
            split_ratio=split_ratio,
            granularity=granularity
        )
        
        # Check if we are using default or uploaded data
        is_default = (filepath == DEFAULT_DATA_PATH)
        results['dataset_info'] = {
            "name": os.path.basename(filepath),
            "is_default": is_default
        }
        
        return jsonify(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file:
        filename = file.filename.lower()
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            return jsonify({"error": "Unsupported file format. Please upload CSV or Excel."}), 400
            
        # Define save path
        save_path = os.path.join(UPLOAD_FOLDER, 'uploaded_dataset.csv')
        
        try:
            if filename.endswith('.csv'):
                # Load CSV and inspect columns
                df = pd.read_csv(file)
            else:
                # Load Excel
                df = pd.read_excel(file)
                
            # Perform basic validation: must have a Date column and a numeric target column
            has_date = False
            for col in df.columns:
                if col.lower() in ['date', 'time', 'timestamp', 'order_date', 'sale_date']:
                    # Rename to standard Date header for simplicity
                    df.rename(columns={col: 'Date'}, inplace=True)
                    has_date = True
                    break
                    
            if not has_date:
                # Fallback: check if first column is date-like
                try:
                    pd.to_datetime(df.iloc[:, 0])
                    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                    has_date = True
                except:
                    pass
                    
            if not has_date:
                return jsonify({"error": "Could not identify a valid Date column. Ensure you have a header named 'Date' or 'Order_Date'."}), 400
                
            # Check for target numeric column
            target_col = None
            for col in df.columns:
                if col.lower() in ['sales_revenue', 'sales', 'revenue', 'value', 'amount', 'total_sales', 'traffic']:
                    target_col = col
                    break
            if not target_col:
                # Find first numeric column that is not Date
                for col in df.columns:
                    if col != 'Date' and pd.api.types.is_numeric_dtype(df[col]):
                        target_col = col
                        break
                        
            if not target_col:
                return jsonify({"error": "Could not find a numeric target column for forecasting. Ensure your file contains sales, revenue, or volume values."}), 400
                
            # Save normalized CSV
            df.to_csv(save_path, index=False)
            set_active_filepath(save_path)
            
            # Immediately run forecast on uploaded file
            model_type = request.form.get('model_type', 'linear')
            horizon = int(request.form.get('horizon', 30))
            split_ratio = float(request.form.get('split_ratio', 0.8))
            granularity = request.form.get('granularity', 'D')
            
            results = preprocess_and_forecast(
                filepath=save_path,
                model_type=model_type,
                horizon=horizon,
                split_ratio=split_ratio,
                granularity=granularity
            )
            results['dataset_info'] = {
                "name": file.filename,
                "is_default": False
            }
            
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 400

@app.route('/api/reset', methods=['POST'])
def reset_dataset():
    if os.path.exists(ACTIVE_DATA_PATH_FILE):
        try:
            os.remove(ACTIVE_DATA_PATH_FILE)
        except:
            pass
            
    # Run default forecast
    try:
        results = preprocess_and_forecast(
            filepath=DEFAULT_DATA_PATH,
            model_type='linear',
            horizon=30,
            split_ratio=0.8,
            granularity='D'
        )
        results['dataset_info'] = {
            "name": os.path.basename(DEFAULT_DATA_PATH),
            "is_default": True
        }
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/export', methods=['GET'])
def export_forecast():
    try:
        model_type = request.args.get('model_type', 'linear')
        horizon = int(request.args.get('horizon', 30))
        split_ratio = float(request.args.get('split_ratio', 0.8))
        granularity = request.args.get('granularity', 'D')
        
        filepath = get_active_filepath()
        
        results = preprocess_and_forecast(
            filepath=filepath,
            model_type=model_type,
            horizon=horizon,
            split_ratio=split_ratio,
            granularity=granularity
        )
        
        forecast_df = pd.DataFrame(results['forecast_data'])
        # Rename columns for user friendliness
        forecast_df.rename(columns={
            "date": "Forecast_Date",
            "prediction": "Point_Forecast_Value",
            "lower_bound": "Lower_Confidence_Bound",
            "upper_bound": "Upper_Confidence_Bound"
        }, inplace=True)
        
        export_path = os.path.join(UPLOAD_FOLDER, 'forecast_export.csv')
        forecast_df.to_csv(export_path, index=False)
        
        return send_file(
            export_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='forecast_predictions.csv'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
