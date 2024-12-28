# app.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from starlette.responses import StreamingResponse
import io

app = FastAPI(title="Vehicle Selling Price Prediction API")
try:
    pipeline = joblib.load("ridge_pipeline.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load the model pipeline: {e}")

class Vehicle(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float
    seats: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str

class Vehicles(BaseModel):
    objects: List[Vehicle]
def prepare_dataframe(data: List[Vehicle]) -> pd.DataFrame:
    try:
        df = pd.DataFrame([item.model_dump() for item in data])
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")

# Endpoint for single prediction
@app.post("/predict_single", summary="Predict selling price for a single vehicle")
def predict_single(vehicle: Vehicle):
    try:
        data = pd.DataFrame([vehicle.model_dump()])
        prediction = pipeline.predict(data)[0]

        return {"selling_price": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch_json", summary="Predict selling prices for multiple vehicles via JSON")
def predict_batch_json(vehicles: Vehicles):
    try:

        df = prepare_dataframe(vehicles.objects)
        predictions = pipeline.predict(df)

        df['predicted_selling_price'] = predictions

        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict_batch_csv", summary="Predict selling prices for multiple vehicles via CSV upload")
async def predict_batch_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        required_columns = ["year", "km_driven", "mileage", "engine", "max_power",
                            "seats", "fuel", "seller_type", "transmission", "owner"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")

        predictions = pipeline.predict(df)
        df['predicted_selling_price'] = predictions

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        response = StreamingResponse(iter([output.getvalue()]),
                                     media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=predictions_{file.filename}"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Welcome to the Vehicle Selling Price Prediction API")
def read_root():
    return {"message": "Welcome to the Vehicle Selling Price Prediction API. Use /docs for API documentation."}