import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import sys
from pathlib import Path

# Add the current directory to sys.path to import local utils
sys.path.insert(0, str(Path(__file__).parent))

from utils.io import load_meta, load_model
from utils.worker_effects import level_to_effect
from utils.data_pipeline import prepare_predict_data

st.set_page_config(page_title="Warehouse Labor Forecaster", layout="wide")

st.title("📊 Warehouse Labor Prediction Dashboard")
st.markdown("""
Upload your **Future Assignment**. 
The system will automatically link reference data (Locations/Products) from your local project directories.
""")

# --- STEP 1: SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
wh_code = st.sidebar.selectbox("Select Warehouse", ["OE", "OF", "RT"])
budget_min = st.sidebar.number_input("Time Budget (min) for Capacity View", value=30)
is_sequenced = st.sidebar.toggle("Use Sequenced Logic", value=False)

st.sidebar.divider()
st.sidebar.header("📁 Input Assignment")
act_file = st.sidebar.file_uploader("Upload Assignment CSV", type=["csv"], 
                                   help="Required columns: ProductID, Quantity, LocationID, WorkCode")

# Hard cap from your notebook
MAX_TASK_TIME = 600 

if act_file:
    # 1. Load and Preview uploaded data
    df_act = pd.read_csv(act_file)
    df_act.columns = df_act.columns.str.strip()
    
    st.subheader("Assignment Preview")
    st.dataframe(df_act.head(), use_container_width=True)

    if "WorkCode" not in df_act.columns:
        st.error("Uploaded file is missing 'WorkCode' column.")
    else:
        # 2. Detect WorkCode
        workcodes = df_act["WorkCode"].astype(str).unique()
        if len(workcodes) > 1:
            st.error(f"Error: Multiple WorkCodes found ({workcodes}). Please use a single-activity file.")
        else:
            target_wc = str(workcodes[0]).split('.')[0]
            st.sidebar.success(f"WorkCode Detected: {target_wc}")

            if st.button("🚀 Run Live Comparison"):
                try:
                    # 3. Load Project Metadata from local models folder
                    meta = load_meta("models", wh_code)
                    if target_wc not in meta["workcodes"]:
                        st.error(f"WorkCode {target_wc} has not been trained for warehouse {wh_code}.")
                        st.stop()

                    wc_encodings = meta["encodings"][target_wc]
                    wc_encodings["train_columns"] = meta["train_columns"][target_wc]

                    # 4. Data Engineering using project directories
                    # This uses the files already in training_data/WH/
                    with st.spinner("Processing data and applying models..."):
                        # Save the upload to a temp path so prepare_predict_data can read it
                        temp_path = "temp_dashboard_queue.csv"
                        df_act.to_csv(temp_path, index=False)
                        
                        df_eng, X_base, _ = prepare_predict_data(
                            temp_path, "training_data", wh_code, 
                            wc_encodings, is_sequenced
                        )

                        # Load model once
                        model = load_model("models", wh_code, target_wc, is_sequenced)
                        
                        # 5. Simulation Loop
                        results = []
                        for level in [1, 2, 3, 4, 5]:
                            X = X_base.copy()
                            # Get performance effect from meta
                            effect = level_to_effect(level, meta["level_medians"].get(target_wc, {}))
                            X["worker_effect"] = effect
                            X = X.reindex(columns=meta["train_columns"][target_wc], fill_value=0)
                            

                            if is_sequenced:
                                # 1. Ensure Travel_Distance handles the first row (NaN -> 0.0)
                                if "Travel_Distance" in X.columns:
                                    X["Travel_Distance"] = X["Travel_Distance"].fillna(0.0)
                                
                                # 2. Ensure same_aisle and same_level handle the first row (NaN -> 0)
                                for col in ["same_aisle", "same_level"]:
                                    if col in X.columns:
                                        X[col] = X[col].fillna(0).astype(int)


                            preds = model.predict(xgb.DMatrix(X))
                            
                            # Goal 1: Total Time
                            total_min = np.sum(preds) / 60
                            
                            # Goal 2: Capacity Simulation
                            tasks_done = 0
                            time_acc = 0
                            budget_sec = budget_min * 60
                            for p in preds:
                                effective_time = min(p, MAX_TASK_TIME)
                                if time_acc + effective_time <= budget_sec:
                                    time_acc += effective_time
                                    tasks_done += 1
                                else: break
                            
                            results.append({
                                "Level": f"Level {level}",
                                "Total Time (min)": round(total_min, 2),
                                "Tasks Completed": tasks_done
                            })

                    # --- STEP 6: VISUALIZE ---
                    res_df = pd.DataFrame(results)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Goal 1: Completion Forecast")
                        fig1 = px.bar(res_df, x="Level", y="Total Time (min)", color="Level",
                                     title=f"Time to finish all {len(df_act)} tasks")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        st.subheader(f"Goal 2: {budget_min} Min Capacity")
                        fig2 = px.bar(res_df, x="Level", y="Tasks Completed", color="Level",
                                     title=f"Tasks finished within time budget")
                        st.plotly_chart(fig2, use_container_width=True)

                    st.divider()
                    st.subheader("Comparison Summary")
                    st.table(res_df)

                except Exception as e:
                    st.error(f"Prediction Pipeline Error: {e}")

else:
    st.info("👈 Please upload your new assignment CSV in the sidebar.")