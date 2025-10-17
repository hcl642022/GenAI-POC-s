import streamlit as st
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv
from vertex_predict import predict_from_vertex
from utils.gcs_helper import download_blob_as_df
 
# Load environment variables
load_dotenv()
 
st.title("Customer 360 Trade Finance Risk Prediction")
 
# --- Input Section ---
st.subheader("Provide Input Data")
 
# Input fields
gcs_path = st.text_input("Enter GCS path (e.g. gs://your-bucket/input.csv)", key="gcs_path_input")
uploaded_file = st.file_uploader("...or upload a CSV file", type=["csv"], key="file_uploader")
 
# Column mapping to match Vertex AI expected names
# IMPORTANT: Use lowercase and underscores for internal representation.
# The actual renaming will happen right before prediction.
internal_column_map = {
    'customer_id': 'customer_id',
    'company_name': 'company_name',
    'total_bg_value': 'total_bg_value',
    'default_count': 'default_count',
    'transaction_count': 'transaction_count',
    'requested_guarantee': 'requested_guarantee',
    'profit_margin': 'profit_margin',
    'debt_to_equity': 'debt_to_equity',
    'credit_score': 'credit_score',
    'previous_bg': 'previous_bg',
    'max_requested_guarantee': 'max_requested_guarantee',
    'on_time_payment_rate': 'on_time_payment_rate',
    'avg_days_late': 'avg_days_late',
    'product_mix': 'product_mix',
    'counter_party_count': 'counter_party_count',
    'cash_flow_gap_days': 'cash_flow_gap_days'
}

# Mapping from internal representation to Vertex AI expected names
vertex_ai_column_map = {
    'customer_id': 'Customer_ID',
    'company_name': 'Company_Name',
    'total_bg_value': 'Total_BG_Value',
    'default_count': 'Default_Count',
    'transaction_count': 'Transaction_Count',
    'requested_guarantee': 'Requested_Guarantee',
    'profit_margin': 'Profit_Margin',
    'debt_to_equity': 'Debt_to_Equity',
    'credit_score': 'Credit_Score',
    'previous_bg': 'Previous_BG',
    'max_requested_guarantee': 'Max_Requested_Guarantee',
    'on_time_payment_rate': 'On_Time_Payment_Rate',
    'avg_days_late': 'Avg_Days_Late',
    'product_mix': 'Product_Mix',
    'counter_party_count': 'Counter_Party_Count',
    'cash_flow_gap_days': 'Cash_Flow_Gap_Days'
}

# --- Load Data ---
if st.button("Load Data"):
    df = None
    if gcs_path:
        try:
            df = download_blob_as_df(gcs_path)
            st.success("File loaded from GCS successfully!")
        except Exception as e:
            st.error(f"Failed to load from GCS: {e}")
    elif uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
 
    if df is not None:
        # ✅ Normalize column names for internal use (lowercase, no spaces)
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        st.write("✅ Columns detected:", df.columns.tolist())
        st.session_state.df = df
    else:
        if 'df' in st.session_state:
            del st.session_state.df
 
# --- Display Data and Predictions ---
if 'df' in st.session_state and st.session_state.df is not None:
    df_display = st.session_state.df.copy() # Use a copy for display to keep original state clean
 
    st.subheader("Uploaded Data Preview")
    st.dataframe(df_display)
 
    # Extract and store Customer IDs separately for easy lookup
    st.subheader("Customer IDs")
    customer_ids_list = df_display.get('customer_id', [])
    if isinstance(customer_ids_list, pd.Series) and not customer_ids_list.empty:
        st.write(customer_ids_list.tolist())
        # Store customer IDs in session state to maintain them
        st.session_state.customer_ids = customer_ids_list.tolist() 
    else:
        st.warning("No 'Customer_Id' column found or it is empty")
        st.session_state.customer_ids = [] # Ensure it's an empty list if not found
 
    # ✅ Convert numeric columns safely
    numeric_cols_internal = ['total_bg_value', 'default_count', 'transaction_count', 'requested_guarantee',
                             'profit_margin', 'debt_to_equity', 'credit_score', 'previous_bg',
                             'max_requested_guarantee', 'on_time_payment_rate', 'avg_days_late',
                             'product_mix', 'counter_party_count', 'cash_flow_gap_days']
    
    for col in numeric_cols_internal:
        if col in df_display.columns:
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
 
    # Replace NaN with None
    df_display = df_display.where(pd.notnull(df_display), None)
 
    # --- Prediction Trigger ---
    if st.button("Run Prediction"):
        try:
            # Create a DataFrame for prediction with Vertex AI-expected column names
            df_to_predict = st.session_state.df.copy() # Use the original internal format of the dataframe
 
            # ✅ Rename columns to the original names expected by Vertex AI
            df_to_predict.rename(columns=vertex_ai_column_map, inplace=True)
 
            # ✅ Debug: show final column names being sent to Vertex AI
            st.write("📤 Columns sent to Vertex AI:", df_to_predict.columns.tolist())
 
            # Prepare payload
            instances = df_to_predict.to_dict(orient="records")
 
            # Call Vertex AI
            response = predict_from_vertex(
                endpoint_id=os.getenv("VERTEX_ENDPOINT_ID"),
                project=os.getenv("PROJECT_ID"),
                location=os.getenv("REGION"),
                instance_list=instances
            )
 
            st.subheader("Prediction Results")
 
            if not response or not hasattr(response, 'predictions') or not response.predictions:
                st.warning("No predictions received from the model.")
            else:
                # Retrieve customer IDs from session state
                ids_to_display = st.session_state.get('customer_ids', [])
 
                for i, prediction in enumerate(response.predictions):
                    predicted_class = "Unknown"
                    confidence = 0.0
 
                    try:
                        # Try Format 1: displayName + confidence
                        if 'displayName' in prediction and 'confidence' in prediction:
                            predicted_class = prediction['displayName']
                            confidence_val = prediction['confidence']
                            confidence = float(confidence_val) if isinstance(confidence_val, (int, float)) else 0.0
                        # Try Format 2: classes + scores
                        elif 'classes' in prediction and 'scores' in prediction:
                            scores = np.array(prediction['scores'])
                            classes = prediction['classes']
                            if scores.size > 0 and len(classes) == scores.size:
                                idx = np.argmax(scores)
                                predicted_class = classes[idx]
                                confidence = float(scores[idx]) if isinstance(scores[idx], (int, float)) else 0.0
                            else:
                                st.warning(f"Record {i+1}: Empty or mismatched scores/classes.")
                        else:
                            st.warning(f"Record {i+1}: Unrecognized prediction format.")
 
                        # Display results using the corresponding customer ID
                        # Use a fallback if i is out of bounds for customer IDs (shouldn't happen with correct logic)
                        cust_id_display = ids_to_display[i] if i < len(ids_to_display) else f"CustID_{i+1}_NotFound"
                        st.write(f"**{cust_id_display}:** Predicted Class = {predicted_class}, Confidence = {round(confidence*100, 2)}%")
 
                    except Exception as parse_err:
                        st.error(f"Prediction parsing failed for record {i+1}: {parse_err}")
 
        except Exception as e:
            st.error(f"Prediction failed: {e}")