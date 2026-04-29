import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import sys
from pathlib import Path

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

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
wh_code = st.sidebar.selectbox("Select Warehouse", ["OE", "OF", "RT"])
budget_min = st.sidebar.number_input("Time Budget (min) for Capacity View", value=30)

st.sidebar.divider()
st.sidebar.header("📁 Input Assignment")
act_file = st.sidebar.file_uploader("Upload Assignment CSV", type=["csv"],
                                    help="Required columns: ProductID, Quantity, LocationID, WorkCode")

MAX_TASK_TIME = 600

if act_file:
    # Load and preview
    df_act = pd.read_csv(act_file)
    df_act.columns = df_act.columns.str.strip()

    st.subheader("Assignment Preview")
    st.dataframe(df_act.head(), use_container_width=True)

    if "WorkCode" not in df_act.columns:
        st.error("Uploaded file is missing 'WorkCode' column.")
    else:
        workcodes = df_act["WorkCode"].astype(str).unique()
        if len(workcodes) > 1:
            st.error(f"Error: Multiple WorkCodes found ({workcodes}). Please use a single-activity file.")
        else:
            target_wc = str(workcodes[0]).split('.')[0]
            st.sidebar.success(f"WorkCode Detected: {target_wc}")

            try:
                # Load metadata
                meta = load_meta("models", wh_code)
                if target_wc not in meta["workcodes"]:
                    st.error(f"WorkCode {target_wc} has not been trained for warehouse {wh_code}.")
                    st.stop()

                wc_encodings = meta["encodings"][target_wc]
                wc_encodings["train_columns"] = meta["train_columns"][target_wc]

                with st.spinner("Processing data and applying models..."):
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
                        df_act.to_csv(tmp, index=False)
                        temp_path = tmp.name

                    df_eng, X_base, _ = prepare_predict_data(
                        temp_path, "training_data", wh_code,
                        wc_encodings, sequenced=False
                    )

                    model = load_model("models", wh_code, target_wc, sequenced=False)
                    try:
                        model_lb = load_model("models", wh_code, target_wc, sequenced=False, lower=True)
                        model_ub = load_model("models", wh_code, target_wc, sequenced=False, upper=True)
                        has_intervals = True
                    except FileNotFoundError:
                        has_intervals = False

                    results = []
                    for level in [1, 2, 3, 4, 5]:
                        X = X_base.copy()
                        effect = level_to_effect(level, meta["level_medians"].get(target_wc, {}))
                        X["worker_effect"] = effect
                        X = X.reindex(columns=meta["train_columns"][target_wc], fill_value=0)

                        dmat  = xgb.DMatrix(X)
                        preds = model.predict(dmat)

                        total_min = round(float(np.sum(preds) / 60), 2)

                        if has_intervals:
                            n          = len(preds)
                            std        = float(np.std(preds))
                            pred_mean  = float(np.mean(preds))
                            lowers     = model_lb.predict(dmat)
                            uppers     = model_ub.predict(dmat)
                            lower_mean = float(np.mean(lowers))
                            upper_mean = float(np.mean(uppers))
                            lb_width   = np.sqrt(n) * (pred_mean - lower_mean) + std
                            ub_width   = np.sqrt(n) * (upper_mean - pred_mean) + std
                            total_sec  = float(np.sum(preds))
                            lower_min  = round(max(0.0, total_sec - lb_width) / 60, 2)
                            upper_min  = round((total_sec + ub_width) / 60, 2)
                        else:
                            lower_min = None
                            upper_min = None

                        tasks_done = 0
                        time_acc = 0.0
                        budget_sec = budget_min * 60
                        for p in preds:
                            effective_time = min(p, MAX_TASK_TIME)
                            if time_acc + effective_time <= budget_sec:
                                time_acc += effective_time
                                tasks_done += 1
                            else:
                                break

                        results.append({
                            "Level":             f"Level {level}",
                            "Total Time (min)":  total_min,
                            "Lower Bound (min)": lower_min,
                            "Upper Bound (min)": upper_min,
                            "Tasks Completed":   tasks_done,  # renamed below
                        })

                # --- VISUALIZE ---
                res_df = pd.DataFrame(results)
                colors = px.colors.qualitative.Plotly
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Goal 1: Completion Forecast")
                    if has_intervals:
                        fig1 = go.Figure()
                        for i, row in res_df.iterrows():
                            total = row["Total Time (min)"]
                            lo    = row["Lower Bound (min)"]
                            hi    = row["Upper Bound (min)"]
                            hover = (
                                f"<b>{row['Level']}</b><br>"
                                f"Predicted: {total:.2f} min<br>"
                                f"95% Interval: {lo:.2f} – {hi:.2f} min"
                            )
                            fig1.add_trace(go.Bar(
                                x=[row["Level"]],
                                y=[total],
                                name=row["Level"],
                                marker_color=colors[i % len(colors)],
                                text=[f"{total:.2f} min<br><sub>[{lo:.2f} – {hi:.2f}]</sub>"],
                                textposition="outside",
                                hovertemplate=hover + "<extra></extra>",
                            ))
                        y_max = res_df["Upper Bound (min)"].max()
                        fig1.update_layout(
                            title=f"Time to finish all {len(df_act)} tasks",
                            uniformtext_minsize=8,
                            uniformtext_mode="hide",
                            yaxis=dict(range=[0, y_max * 1.025]),
                        )
                    else:
                        fig1 = px.bar(res_df, x="Level", y="Total Time (min)", color="Level",
                                      title=f"Time to finish all {len(df_act)} tasks",
                                      text_auto=".2f")
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    st.subheader(f"Goal 2: {budget_min} Min Capacity")
                    fig2 = go.Figure()
                    for i, row in res_df.iterrows():
                        fig2.add_trace(go.Bar(
                            x=[row["Level"]],
                            y=[row["Tasks Completed"]],
                            name=row["Level"],
                            marker_color=colors[i % len(colors)],
                            text=[row["Tasks Completed"]],
                            textposition="outside",
                            hovertemplate=f"<b>{row['Level']}</b><br>Tasks: {row['Tasks Completed']}<extra></extra>",
                        ))
                    y_max2 = res_df["Tasks Completed"].max()
                    fig2.update_layout(
                        title=f"Tasks finished within {budget_min} min budget",
                        yaxis=dict(range=[0, y_max2 * 1.15]),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.divider()
                st.subheader("Comparison Summary")

                col_tasks = f"Tasks Completed in {budget_min} min"
                if has_intervals:
                    display_df = res_df[["Level", "Total Time (min)", "Lower Bound (min)", "Upper Bound (min)", "Tasks Completed"]].copy()
                    display_df["Interval (min)"] = display_df.apply(
                        lambda r: f"{r['Lower Bound (min)']:.2f} – {r['Upper Bound (min)']:.2f}", axis=1
                    )
                    display_df = display_df[["Level", "Total Time (min)", "Interval (min)", "Tasks Completed"]]
                    display_df = display_df.rename(columns={"Tasks Completed": col_tasks})
                else:
                    display_df = res_df[["Level", "Total Time (min)", "Tasks Completed"]].copy()
                    display_df = display_df.rename(columns={"Tasks Completed": col_tasks})

                for col in display_df.select_dtypes(include="number").columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

                st.table(display_df)

                if not has_intervals:
                    st.caption("⚠️ LB/UB models not found — prediction intervals unavailable. Train quantile models to enable.")

            except Exception as e:
                st.error(f"Prediction Pipeline Error: {e}")

else:
    st.info("👈 Please upload your new assignment CSV in the sidebar.")
