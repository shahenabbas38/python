from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import random
import json
from datetime import datetime
from typing import Optional

app = FastAPI(title="Nutrition AI Service")

# --- الإعدادات والدوال الأساسية ---

def get_dataset_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "FINAL FOOD DATASET"),
        os.path.join(os.path.dirname(base_dir), "FINAL FOOD DATASET"),
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    return None

COLUMN_MAP = {
    'name': ["food", "Unnamed: 1", "Name", "food_name"],
    'cal': ["Caloric Value", "Calories", "Energy", "calories"],
    'prot': ["Protein", "protein"],
    'carb': ["Carbohydrates", "Carbs", "carb"],
    'fat': ["Fat", "fat"]
}

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def calculate_age(dob_str):
    try:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                dob = datetime.strptime(dob_str, fmt)
                break
            except: continue
        else: return 30
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except: return 30

def calculate_bmr(weight, height, gender, age):
    if str(gender).lower() == "male":
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    return (10 * weight) + (6.25 * height) - (5 * age) - 161

# --- نموذج البيانات المتوقع من Laravel ---

class PatientProfile(BaseModel):
    full_name: str
    weight_kg: float
    height_cm: float
    gender: str
    dob: str
    primary_condition: Optional[str] = "NONE"

# --- نقطة النهاية (Endpoint) ---

@app.post("/recommend")
async def get_recommendations(profile: PatientProfile):
    try:
        DATASET_DIR = get_dataset_path()
        if not DATASET_DIR:
            raise HTTPException(status_code=500, detail="Dataset folder not found")

        files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(".csv")]
        if not files:
            raise HTTPException(status_code=500, detail="No CSV files found")

        df = pd.concat([pd.read_csv(os.path.join(DATASET_DIR, f)) for f in files], ignore_index=True)
        cols = {k: find_col(df, v) for k, v in COLUMN_MAP.items()}

        if not cols['name'] or not cols['cal']:
            raise HTTPException(status_code=500, detail="Required columns missing in CSV")

        for k, c in cols.items():
            if c and k != 'name':
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # الحسابات
        age = calculate_age(profile.dob)
        daily_calories = calculate_bmr(profile.weight_kg, profile.height_cm, profile.gender, age) * 1.2
        max_meal_cal = daily_calories / 3
        condition = profile.primary_condition.upper()

        # التصفية بناءً على الحالة الصحية
        filtered = df[df[cols['cal']] <= max_meal_cal].copy()
        if "DIABETES" in condition:
            if cols['carb']: filtered = filtered[filtered[cols['carb']] <= 25]
        elif "OBESITY" in condition or "HEART" in condition:
            if cols['fat']: filtered = filtered[filtered[cols['fat']] <= 10]

        def create_meal_list(data, meal_type):
            if data.empty: return []
            items = data.sample(n=min(5, len(data)))
            return [{
                "food_name": str(row[cols['name']]),
                "calories": round(float(row[cols['cal']]), 2),
                "protein": round(float(row[cols['prot']]), 2) if cols['prot'] else 0,
                "carbohydrates": round(float(row[cols['carb']]), 2) if cols['carb'] else 0,
                "fat": round(float(row[cols['fat']]), 2) if cols['fat'] else 0,
                "meal_type": meal_type
            } for _, row in items.iterrows()]

        return {
            "status": "success",
            "patient_info": {
                "full_name": profile.full_name,
                "daily_calories": round(daily_calories, 2),
                "condition": condition
            },
            "meals": {
                "breakfast": create_meal_list(filtered, "BREAKFAST"),
                "lunch": create_meal_list(filtered, "LUNCH"),
                "dinner": create_meal_list(filtered, "DINNER")
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Railway سيستخدم PORT من المتغيرات البيئية
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)