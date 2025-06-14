from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import io
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Sales Forecasting System",
    description="A predictive AI tool for accurate sales forecasting using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ForecastRequest(BaseModel):
    periods: int = 30
    include_history: bool = False
    confidence_interval: float = 0.95

class ForecastResponse(BaseModel):
    forecast_data: List[Dict[str, Any]]
    metrics: Dict[str, float]
    chart_data: Dict[str, Any]
    model_info: Dict[str, str]

class DataUploadResponse(BaseModel):
    message: str
    data_shape: tuple
    columns: List[str]
    date_range: Dict[str, str]

# Global variables to store data and model
df_data = None
model = None
df_prophet = None

class SalesForecastingEngine:
    def __init__(self):
        self.model = None
        self.df_data = None
        self.df_prophet = None
        self.is_trained = False
        
    def preprocess_data(self, df_data):
        """Preprocess the uploaded data"""
        try:
            # Convert date column to datetime
            df_data.date = pd.to_datetime(df_data.date)
            
            # Extract date features
            df_data['year'] = df_data['date'].dt.year
            df_data['month'] = df_data['date'].dt.month
            df_data['week'] = df_data['date'].dt.isocalendar().week
            df_data['quarter'] = df_data['date'].dt.quarter
            df_data['day_of_week'] = df_data['date'].dt.day_name()
            
            # Handle missing values
            if 'dcoilwtico' in df_data.columns:
                df_data['dcoilwtico'] = df_data['dcoilwtico'].bfill()
            if 'transactions' in df_data.columns:
                df_data.transactions = df_data.transactions.replace(np.nan, 0)
            if 'holiday_type' in df_data.columns:
                df_data['holiday_type'] = df_data['holiday_type'].replace(np.nan, 'Working Day')
            if 'transferred' in df_data.columns:
                df_data['transferred'] = df_data['transferred'].replace(np.nan, False)
            
            # Fill other string columns
            string_cols = ['locale', 'locale_name', 'description']
            for col in string_cols:
                if col in df_data.columns:
                    df_data[col] = df_data[col].replace(np.nan, '')
            
            # Encode categorical variables
            non_numerical_cols = [col for col in df_data.columns if df_data[col].dtype == 'object']
            for feature in non_numerical_cols:
                if feature != 'date':  # Don't encode the date column
                    df_data[feature] = LabelEncoder().fit_transform(df_data[feature].astype(str))
            
            self.df_data = df_data
            return df_data
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(e)}")
    
    def prepare_prophet_data(self):
        """Prepare data for Prophet model"""
        try:
            # Check for required columns
            required_cols = {'date', 'sales'}
            if not required_cols.issubset(self.df_data.columns):
                raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_cols}")

            # Build aggregation dict only for columns that exist
            agg_dict = {'sales': 'mean'}
            for col in ['onpromotion', 'transactions', 'dcoilwtico']:
                if col in self.df_data.columns:
                    agg_dict[col] = 'mean'

            df_prophet = self.df_data.groupby('date').agg(agg_dict).reset_index()
            df_prophet = df_prophet.rename(columns={'date': 'ds', 'sales': 'y'})
            df_prophet = df_prophet.dropna(subset=['y'])
            self.df_prophet = df_prophet
            return df_prophet
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prophet data preparation failed: {str(e)}")
    
    def train_model(self):
        """Train the Prophet model"""
        try:
            if self.df_prophet is None:
                raise ValueError("Prophet data not prepared")
            
            # Split data for training (use 80% for training)
            split_date = self.df_prophet['ds'].quantile(0.8)
            df_train = self.df_prophet[self.df_prophet['ds'] <= split_date]
            
            # Initialize Prophet model
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )
            
            # Add regressors if available
            if 'onpromotion' in self.df_prophet.columns:
                self.model.add_regressor('onpromotion')
            if 'transactions' in self.df_prophet.columns:
                self.model.add_regressor('transactions')
            if 'dcoilwtico' in self.df_prophet.columns:
                self.model.add_regressor('dcoilwtico')
            
            # Fit the model
            self.model.fit(df_train)
            self.is_trained = True
            
            return {"status": "Model trained successfully", "training_samples": len(df_train)}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
    
    def generate_forecast(self, periods=30, include_history=False):
        """Generate sales forecast"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)
            
            # Add regressor values for future dates
            # For simplicity, use mean values for future periods
            if 'onpromotion' in self.df_prophet.columns:
                future['onpromotion'] = self.df_prophet['onpromotion'].mean()
            if 'transactions' in self.df_prophet.columns:
                future['transactions'] = self.df_prophet['transactions'].mean()
            if 'dcoilwtico' in self.df_prophet.columns:
                future['dcoilwtico'] = self.df_prophet['dcoilwtico'].mean()
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Calculate metrics on validation data
            split_date = self.df_prophet['ds'].quantile(0.8)
            df_valid = self.df_prophet[self.df_prophet['ds'] > split_date]
            forecast_valid = forecast[forecast['ds'] > split_date]
            
            # Merge for comparison
            comparison = forecast_valid.merge(df_valid[['ds', 'y']], on='ds', how='inner')
            
            metrics = {}
            if len(comparison) > 0:
                metrics['mae'] = mean_absolute_error(comparison['y'], comparison['yhat'])
                metrics['rmse'] = np.sqrt(mean_squared_error(comparison['y'], comparison['yhat']))
                metrics['mape'] = np.mean(np.abs((comparison['y'] - comparison['yhat']) / comparison['y'])) * 100
            
            # Prepare response data
            if include_history:
                forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
            else:
                future_forecast = forecast.tail(periods)
                forecast_data = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
            
            return forecast_data, metrics, forecast
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

# Initialize the forecasting engine
forecasting_engine = SalesForecastingEngine()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("ui.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/upload-data", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload and preprocess sales data"""
    print("Uploading the data")
    try:
        # Read uploaded file
        contents = await file.read()
        print("Got the file")
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        print("Read the file")
        # Preprocess the data
        processed_df = forecasting_engine.preprocess_data(df)
        print("Preprocessed the data")
        # Prepare Prophet data
        forecasting_engine.prepare_prophet_data()
        print("Prepared the prophet data")
        date_range = {
            "start_date": str(processed_df['date'].min()),
            "end_date": str(processed_df['date'].max())
        }
        print("Returned the data")
        return DataUploadResponse(
            message="Data uploaded and preprocessed successfully",
            data_shape=processed_df.shape,
            columns=processed_df.columns.tolist(),
            date_range=date_range
        )
        
    except Exception as e:
        import traceback
        print(f"ERROR: Exception in upload_data: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e) or 'Unknown processing error'}")

@app.post("/train-model")
async def train_model():
    """Train the forecasting model"""
    try:
        if forecasting_engine.df_prophet is None:
            raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first.")
        
        result = forecasting_engine.train_model()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/generate-forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """Generate sales forecast"""
    try:
        if not forecasting_engine.is_trained:
            raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
        
        forecast_data, metrics, full_forecast = forecasting_engine.generate_forecast(
            periods=request.periods,
            include_history=request.include_history
        )
        
        # Create interactive chart
        fig = go.Figure()
        
        # Add actual data if including history
        if request.include_history and forecasting_engine.df_prophet is not None:
            fig.add_trace(go.Scatter(
                x=forecasting_engine.df_prophet['ds'],
                y=forecasting_engine.df_prophet['y'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='blue')
            ))
        
        # Add forecast
        forecast_df = pd.DataFrame(forecast_data)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='red')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig.update_layout(
            title='Sales Forecast',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        chart_data = json.loads(fig.to_json())
        
        return ForecastResponse(
            forecast_data=forecast_data,
            metrics=metrics,
            chart_data=chart_data,
            model_info={
                "model_type": "Prophet",
                "forecast_periods": str(request.periods),
                "confidence_interval": str(request.confidence_interval)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": forecasting_engine.is_trained,
        "data_loaded": forecasting_engine.df_data is not None
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the current model and data"""
    return {
        "model_trained": forecasting_engine.is_trained,
        "data_loaded": forecasting_engine.df_data is not None,
        "data_shape": forecasting_engine.df_data.shape if forecasting_engine.df_data is not None else None,
        "date_range": {
            "start": str(forecasting_engine.df_data['date'].min()) if forecasting_engine.df_data is not None else None,
            "end": str(forecasting_engine.df_data['date'].max()) if forecasting_engine.df_data is not None else None
        } if forecasting_engine.df_data is not None else None
    }

@app.post("/forecast-image")
async def forecast_image(periods: int = 30, include_history: bool = True):
    if not forecasting_engine.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")

    # Generate forecast
    forecast_data, metrics, y_pred = forecasting_engine.generate_forecast(
        periods=periods,
        include_history=True  # Always get full history for plotting
    )

    # Prepare data
    train = forecasting_engine.df_prophet.copy()
    forecast_df = y_pred.copy()
    forecast_start = forecast_df['ds'].iloc[-periods]  # The first forecasted date

    # Plot
    f, ax = plt.subplots(1)
    f.set_figheight(10)
    f.set_figwidth(22)

    # Plot actuals up to forecast start
    train_hist = train[train['ds'] < forecast_start]
    ax.plot(train_hist['ds'], train_hist['y'], color='darkorange', label='Observed data points')

    # Plot forecast (yhat) from forecast start onward
    forecast_future = forecast_df[forecast_df['ds'] >= forecast_start]
    ax.plot(forecast_future['ds'], forecast_future['yhat'], color='royalblue', label='Forecast')

    # Uncertainty interval
    ax.fill_between(
        forecast_future['ds'],
        forecast_future['yhat_lower'],
        forecast_future['yhat_upper'],
        color='royalblue',
        alpha=0.2,
        label='Uncertainty interval'
    )

    # Vertical line at forecast start
    ax.axvline(forecast_start, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Forecast Start')

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.set_title('Average Sales per Day')
    ax.legend()

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        plt.savefig(tmpfile.name, bbox_inches='tight')
        plt.close(f)
        filename = tmpfile.name

    return FileResponse(
        filename,
        media_type="image/png",
        filename="forecast.png",
        headers={"Content-Disposition": "attachment; filename=forecast.png"}
    )
    # Optionally, clean up the file after sending (if needed)
    # os.remove(filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)