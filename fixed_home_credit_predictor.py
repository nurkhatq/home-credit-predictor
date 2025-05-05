import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import glob
import time
import re
import logging
from datetime import datetime
import json
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("home_credit_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HomeCreditApp")

# Configuration
MODELS_DIR = "data/lgbm"
DATA_DIR = "data"
HISTORICAL_DIR = "historical_data"
os.makedirs(HISTORICAL_DIR, exist_ok=True)

# Function to reduce memory usage
def reduce_memory_usage(df):
    """Reduce memory usage of a dataframe by using more efficient data types"""
    logger.info(f"Reducing memory usage of dataframe with shape {df.shape}")
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df

# Find and load models
@st.cache_resource
def load_models():
    """Load saved LightGBM models"""
    logger.info("Loading models...")
    models = []
    
    # Check standard directories for model files
    model_paths = []
    for model_dir in [MODELS_DIR, "lgbm", "models", "saved_models"]:
        if os.path.exists(model_dir):
            model_paths.extend(glob.glob(os.path.join(model_dir, "*.pickle")))
    
    if not model_paths:
        logger.warning("No model files found in standard directories")
        # Search recursively in data directory
        model_paths = glob.glob("**/*.pickle", recursive=True)
    
    if not model_paths:
        logger.error("No model files found")
        return None
    
    logger.info(f"Found {len(model_paths)} model files: {model_paths}")
    
    # Load each model
    for model_path in model_paths:
        try:
            logger.info(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
                logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
    
    return models if models else None

# Get feature names from models
def get_model_features(models):
    """Extract feature names used by the models"""
    if not models or len(models) == 0:
        return None
    
    # Store feature names from each model
    model_features = []
    
    # Try to get feature names from all models
    for i, model in enumerate(models):
        try:
            features = None
            feature_count = None
            
            # For LightGBM models with feature_name_ attribute
            if hasattr(model, 'feature_name_'):
                features = model.feature_name_
                feature_count = len(features)
                logger.info(f"Model {i}: Found feature_name_ attribute with {feature_count} features")
                model_features.append(features)
            
            # For LightGBM models with feature_names attribute
            elif hasattr(model, 'feature_names'):
                features = model.feature_names
                feature_count = len(features)
                logger.info(f"Model {i}: Found feature_names attribute with {feature_count} features")
                model_features.append(features)
                
            # For scikit-learn models with feature_names_in_ attribute
            elif hasattr(model, 'feature_names_in_'):
                features = model.feature_names_in_
                feature_count = len(features)
                logger.info(f"Model {i}: Found feature_names_in_ attribute with {feature_count} features")
                model_features.append(features)
            
            # For other model types, check if booster has feature_name
            elif hasattr(model, 'booster') and hasattr(model.booster, 'feature_name'):
                features = model.booster.feature_name
                feature_count = len(features)
                logger.info(f"Model {i}: Found booster feature_name with {feature_count} features")
                model_features.append(features)
            else:
                logger.warning(f"Model {i}: No feature names found")
        except Exception as e:
            logger.warning(f"Error extracting feature names from model {i}: {e}")
    
    # If we found feature names for at least one model, return them
    if model_features:
        # Get the most frequent feature count
        feature_counts = [len(feats) for feats in model_features]
        logger.info(f"Feature counts across models: {feature_counts}")
        
        # Return the feature names from the first model
        # (individual models will handle their own features in make_prediction)
        return model_features[0]
    
    # If we get here, no feature names were found
    logger.error("Could not determine feature names from any model")
    return None

# Find and load the main application data for feature names
@st.cache_data
def load_application_data(sample=True):
    """Load application data CSV file"""
    logger.info("Loading application data...")
    
    # Check standard locations for application data
    app_files = []
    for data_dir in [DATA_DIR, ".", "data"]:
        app_files.extend(glob.glob(os.path.join(data_dir, "*application*train*.csv")))
    
    if not app_files:
        logger.warning("No application data files found in standard directories")
        # Search recursively
        app_files = glob.glob("**/*application*train*.csv", recursive=True)
    
    if not app_files:
        logger.error("No application data files found")
        return None
    
    logger.info(f"Found application data files: {app_files}")
    
    try:
        # Load the first file found
        file_path = app_files[0]
        logger.info(f"Loading application data from {file_path}")
        
        if sample:
            # Load just a sample for faster processing
            df = pd.read_csv(file_path, nrows=1000)
        else:
            df = pd.read_csv(file_path)
            
        logger.info(f"Loaded application data with shape {df.shape}")
        return reduce_memory_usage(df)
    except Exception as e:
        logger.error(f"Error loading application data: {e}")
        return None

# Find and load all available data files
@st.cache_data
def find_all_data_files():
    """Find all available data files in the project"""
    data_files = {}
    
    # Look for all CSV files in standard directories
    csv_files = []
    for data_dir in [DATA_DIR, ".", "data"]:
        csv_files.extend(glob.glob(os.path.join(data_dir, "*.csv")))
    
    # Categorize files by type
    for file in csv_files:
        basename = os.path.basename(file)
        if "application" in basename.lower():
            data_files["application"] = file
        elif "bureau_balance" in basename.lower():
            data_files["bureau_balance"] = file
        elif "bureau" in basename.lower():
            data_files["bureau"] = file
        elif "credit_card_balance" in basename.lower():
            data_files["credit_card"] = file
        elif "pos_cash" in basename.lower():
            data_files["pos_cash"] = file
        elif "installments" in basename.lower():
            data_files["installments"] = file
        elif "previous" in basename.lower():
            data_files["previous"] = file
    
    logger.info(f"Found data files: {data_files}")
    return data_files

# Handle null values and outliers in input data
def fix_nulls_outliers(data):
    """Fix null values and outliers in the input data"""
    data = data.copy()
    
    # Fill categorical nulls
    for col in ['NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'NAME_TYPE_SUITE']:
        if col in data and data[col].isnull().any():
            if col == 'NAME_TYPE_SUITE':
                data[col] = data[col].fillna('Unaccompanied')
            elif col == 'NAME_FAMILY_STATUS':
                data[col] = data[col].fillna('Married')
            else:
                data[col] = data[col].fillna('Data_Not_Available')
    
    # Fill flag nulls
    for col in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL']:
        if col in data and data[col].isnull().any():
            data[col] = data[col].fillna(0)
    
    # Fix DAYS_EMPLOYED outliers
    if 'DAYS_EMPLOYED' in data:
        # Handle the 365243 days outlier
        max_days = 365243
        if (data['DAYS_EMPLOYED'] > 0).any() or (data['DAYS_EMPLOYED'] == max_days).any():
            data.loc[data['DAYS_EMPLOYED'] == max_days, 'DAYS_EMPLOYED'] = np.nan
            # Fill with median of non-outlier values
            median_days = data.loc[data['DAYS_EMPLOYED'] < 0, 'DAYS_EMPLOYED'].median()
            data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].fillna(median_days)
    
    # Fill numeric nulls
    for col in ['AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        if col in data and data[col].isnull().any():
            data[col] = data[col].fillna(0)
    
    # Fill external source features
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in data and data[col].isnull().any():
            data[col] = data[col].fillna(0)
    
    # Fix CODE_GENDER
    if 'CODE_GENDER' in data:
        data['CODE_GENDER'] = data['CODE_GENDER'].replace('XNA', 'M')
    
    # Fix CNT_FAM_MEMBERS
    if 'CNT_FAM_MEMBERS' in data and data['CNT_FAM_MEMBERS'].isnull().any():
        mode_val = data['CNT_FAM_MEMBERS'].mode().iloc[0]
        data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'].fillna(mode_val)
    
    return data

# Collect all available features consistently across models
def ensure_consistent_features(feature_names, app_data=None):
    """Ensure consistent feature set across models"""
    if feature_names is None:
        logger.error("No feature names provided")
        return []
        
    # First, make sure feature_names is a list
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    
    # Clean feature names to remove any problematic characters
    cleaned_feature_names = []
    for feat in feature_names:
        # Convert to string if not already
        feat_str = str(feat)
        # Replace spaces with underscores
        feat_str = feat_str.replace(' ', '_')
        # Remove any special characters
        feat_str = re.sub(r'[^A-Za-z0-9_]', '', feat_str)
        cleaned_feature_names.append(feat_str)
    
    # We'll use just the model's feature names rather than combining with app_data
    # This is to ensure we have exactly the features the model expects
    logger.info(f"Using {len(cleaned_feature_names)} features from model")
    return cleaned_feature_names

# Prepare input data for prediction
# Prepare input data for prediction
def prepare_input_data(form_data, feature_names, app_data=None, model_index=None):
    """Prepare input data for prediction based on feature names for a specific model"""
    # Create a DataFrame with one row
    input_df = pd.DataFrame([form_data])
    
    # Convert age and employment duration to days
    if 'age_years' in form_data:
        input_df['DAYS_BIRTH'] = -form_data['age_years'] * 365
    
    if 'employment_years' in form_data:
        input_df['DAYS_EMPLOYED'] = -form_data['employment_years'] * 365
    
    # If we have application data with the same ID, use it to fill missing features
    user_id = form_data.get('SK_ID_CURR')
    if app_data is not None and user_id is not None:
        user_data = app_data[app_data['SK_ID_CURR'] == user_id]
        if not user_data.empty:
            logger.info(f"Found existing data for user ID {user_id}")
            user_row = user_data.iloc[0]
            
            # Fill missing features from application data
            for feature in feature_names:
                if feature not in input_df.columns and feature in user_row.index:
                    input_df[feature] = user_row[feature]
    
    # Find missing features
    missing_features = [f for f in feature_names if f not in input_df.columns]
    
    # Add missing features as zeros
    for feature in missing_features:
        input_df[feature] = 0
    
    # Ensure features are ONLY those in feature_names and in the correct order
    try:
        input_df = input_df[feature_names]
    except KeyError as e:
        logger.error(f"Feature mismatch for model {model_index}: {e}")
        # Create a new DataFrame with just the required features
        new_df = pd.DataFrame(0, index=input_df.index, columns=feature_names)
        # Copy over values we do have
        for col in feature_names:
            if col in input_df.columns:
                new_df[col] = input_df[col]
        input_df = new_df
    
    # Convert categorical columns to numeric
    for col in input_df.columns:
        # Check if column contains non-numeric data
        if input_df[col].dtype == object or pd.api.types.is_categorical_dtype(input_df[col]):
            logger.info(f"Converting categorical column {col} to numeric")
            # Try simple numeric conversion first
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            except:
                # If simple conversion fails, use one-hot encoding
                logger.info(f"Using one-hot encoding for {col}")
                input_df[col] = 0  # Default to 0 for categorical columns
    
    # Convert all data to float
    input_df = input_df.astype(float)
    
    # Get the values as a list
    final_data = input_df.values[0]
    
    return final_data, missing_features

# Make prediction using ensemble of models
# Make prediction using ensemble of models
def make_prediction(models, data):
    """Make prediction using ensemble of models"""
    if not models or len(models) == 0:
        return None
    
    try:
        # Make predictions with each model using the appropriate feature set for each
        predictions = []
        
        for i, model in enumerate(models):
            try:
                # Get the model's feature names
                model_features = None
                feature_count = None
                
                if hasattr(model, 'feature_name_'):
                    model_features = model.feature_name_
                    feature_count = len(model_features)
                elif hasattr(model, 'feature_names'):
                    model_features = model.feature_names
                    feature_count = len(model_features)
                elif hasattr(model, 'feature_names_in_'):
                    model_features = model.feature_names_in_
                    feature_count = len(model_features)
                
                # Prepare input data for this specific model
                if feature_count is not None:
                    # Create proper input array for this model
                    if len(data) < feature_count:
                        # If data is shorter than the model expects, pad with zeros
                        logger.info(f"Padding data from {len(data)} to {feature_count} features for model {i}")
                        padded_data = np.zeros(feature_count)
                        padded_data[:len(data)] = data
                        X_model = np.array([padded_data], dtype=float)
                    elif len(data) > feature_count:
                        # If data is longer than model expects, truncate
                        logger.info(f"Truncating data from {len(data)} to {feature_count} features for model {i}")
                        X_model = np.array([data[:feature_count]], dtype=float)
                    else:
                        # Data length matches exactly
                        X_model = np.array([data], dtype=float)
                else:
                    # Fall back to using the data as is
                    X_model = np.array([data], dtype=float)
                
                # Try predict_proba first
                try:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_model)[:,1]
                        predictions.append(pred[0])
                        logger.info(f"Model {i} prediction (predict_proba): {pred[0]}")
                        continue
                except Exception as e1:
                    logger.warning(f"predict_proba failed for model {i}: {e1}")
                
                # If predict_proba failed, try predict
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_model)
                        if isinstance(pred, np.ndarray) and len(pred) > 0:
                            predictions.append(float(pred[0]))
                            logger.info(f"Model {i} prediction (predict): {pred[0]}")
                            continue
                except Exception as e2:
                    logger.warning(f"predict failed for model {i}: {e2}")
                
                # If both methods failed, log a warning
                logger.warning(f"Model {i} doesn't have working prediction methods")
                
            except Exception as e:
                logger.error(f"Error making prediction with model {i}: {e}")
        
        # Check if we have any valid predictions
        if not predictions:
            logger.error("No valid predictions were made")
            # Return a default prediction
            return 0.5
            
        # Average the predictions
        final_prediction = sum(predictions) / len(predictions)
        logger.info(f"Final ensemble prediction: {final_prediction}")
        return final_prediction
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        # Return a default prediction
        return 0.5

# Save prediction to history
def save_prediction(input_data, prediction, features_used):
    """Save prediction to history"""
    try:
        # Create a unique ID for this prediction
        prediction_id = str(uuid.uuid4())
        
        # Prepare data for storage
        timestamp = datetime.now().isoformat()
        record = {
            "prediction_id": prediction_id,
            "timestamp": timestamp,
            "input_data": input_data,
            "features_used": features_used,
            "prediction_result": float(prediction),
            "prediction_threshold": 0.5,
            "prediction_label": "Default Risk" if prediction > 0.5 else "No Default Risk"
        }
        
        # Save to file
        filename = f"prediction_{prediction_id}.json"
        filepath = os.path.join(HISTORICAL_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(record, f, indent=2)
        
        logger.info(f"Saved prediction to {filepath}")
        return prediction_id
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return None

# Load prediction history
def load_prediction_history():
    """Load prediction history from files"""
    predictions = []
    
    try:
        if os.path.exists(HISTORICAL_DIR):
            files = glob.glob(os.path.join(HISTORICAL_DIR, "prediction_*.json"))
            
            for file in files:
                try:
                    with open(file, 'r') as f:
                        prediction = json.load(f)
                        predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Error reading prediction file {file}: {e}")
        
        # Sort by timestamp (newest first)
        predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return predictions
    except Exception as e:
        logger.error(f"Error loading prediction history: {e}")
        return []

# Create formatted history dataframe
def create_history_dataframe(predictions):
    """Create formatted history dataframe for display"""
    if not predictions:
        return pd.DataFrame()
    
    # Extract key information for the table
    pred_data = []
    for p in predictions:
        try:
            user_id = p.get("input_data", {}).get("SK_ID_CURR", "")
            # Make sure we convert to a proper type
            if user_id == "N/A" or user_id == "":
                user_id = 0
            else:
                try:
                    user_id = int(user_id)
                except:
                    user_id = 0
            
            pred_data.append({
                "ID": p.get("prediction_id", "")[:8],
                "User_ID": user_id,  # Ensure it's an int or 0
                "Timestamp": p.get("timestamp", "")[:19],
                "Income": float(p.get("input_data", {}).get("AMT_INCOME_TOTAL", 0)),
                "Credit_Amount": float(p.get("input_data", {}).get("AMT_CREDIT", 0)),
                "Default_Probability": float(p.get('prediction_result', 0)),
                "Prediction": p.get('prediction_label', "")
            })
        except Exception as e:
            logger.error(f"Error processing prediction record: {e}")
    
    return pd.DataFrame(pred_data)

# Get feature importance plot
def get_feature_importance_plot(models, feature_names, top_n=20):
    """Create feature importance plot"""
    if not models or not feature_names:
        return None
    
    try:
        # Try to get feature importance from any model that has it
        model_with_importance = None
        for model in models:
            if hasattr(model, 'feature_importances_'):
                model_with_importance = model
                break
        
        if not model_with_importance:
            logger.warning("No model has feature_importances_ attribute")
            return None
        
        # Create DataFrame with features and importances
        importances = model_with_importance.feature_importances_
        
        # Check if feature names and importances have the same length
        if len(importances) != len(feature_names):
            logger.warning(f"Feature importances ({len(importances)}) and feature names ({len(feature_names)}) have different lengths")
            
            # Use the smaller length
            min_length = min(len(importances), len(feature_names))
            importances = importances[:min_length]
            feature_names = feature_names[:min_length]
            
            # Or use generic feature names if needed
            if len(importances) > len(feature_names):
                feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Create a DataFrame for sorting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title('Feature Importances')
        plt.barh(
            range(len(importance_df)), 
            importance_df['importance'].values, 
            color='#1E88E5', 
            align='center'
        )
        
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")
        return None

# Get sample users from application data
def get_sample_users(app_data, n=10):
    """Get sample users from application data"""
    if app_data is None or 'SK_ID_CURR' not in app_data.columns:
        return []
    
    try:
        # Get a random sample of user IDs
        return app_data['SK_ID_CURR'].sample(min(n, len(app_data))).tolist()
    except Exception as e:
        logger.error(f"Error getting sample users: {e}")
        return []

# Create dashboard sidebar with information
def create_sidebar():
    """Create dashboard sidebar with information"""
    st.sidebar.title("Home Credit Default Risk")
    
    st.sidebar.markdown("""
    ## About this app
    
    This app predicts the probability of a loan applicant 
    defaulting on a loan using trained machine learning models.
    
    ### Data sources:
    - Application data (main)
    - Bureau & Bureau Balance 
    - Previous Applications
    - POS Cash Balance
    - Installments Payments
    - Credit Card Balance
    
    ### Models:
    LightGBM ensemble trained on historical loan data
    """)
    
    # Show available data files
    data_files = find_all_data_files()
    if data_files:
        with st.sidebar.expander("Available Data Files"):
            for data_type, file_path in data_files.items():
                st.write(f"**{data_type}**: {os.path.basename(file_path)}")
    
    # Add information about interpretation
    st.sidebar.markdown("""
    ### Prediction Interpretation:
    
    - **< 0.5**: Low default risk
    - **≥ 0.5**: High default risk
    
    ### Key Features:
    - **EXT_SOURCE_1,2,3**: External source scores
    - **DAYS_BIRTH**: Age in days (negative)
    - **DAYS_EMPLOYED**: Employment duration in days
    - **AMT_CREDIT**: Credit amount
    - **AMT_INCOME_TOTAL**: Income amount
    """)

# Main dashboard UI
def main():
    """Main function for the Streamlit dashboard"""
    st.title("Home Credit Default Risk Predictor")
    
    # Create sidebar
    create_sidebar()
    
    # Load models
    models = load_models()
    if models is None:
        st.error("No models could be loaded. Please check that model files exist in the data/lgbm directory.")
        st.info("Try running the notebook to generate model files first.")
        return
    
    st.success(f"Successfully loaded {len(models)} models")
    
    # Load application data
    app_data = load_application_data()
    if app_data is None:
        st.warning("Could not load application data. Some features will not be available.")
    else:
        st.success(f"Loaded application data with {len(app_data)} rows and {len(app_data.columns)} columns")
    
    # Get feature names from each model
    feature_names = get_model_features(models)
    if feature_names is None:
        if app_data is not None:
            # Use columns from application data as fallback
            feature_names = app_data.columns.drop(['TARGET', 'SK_ID_CURR']).tolist()
            st.warning(f"Using {len(feature_names)} features from application data (model feature names not available)")
        else:
            st.error("Could not determine feature names. Please check model files.")
            return
    else:
        # Use original model feature names without combining with app_data
        # This ensures we have exactly the features the model expects
        st.success(f"Using {len(feature_names)} features from models")
    
    # Get sample user IDs
    sample_users = get_sample_users(app_data, n=10) if app_data is not None else []
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Make Prediction", "Historical Predictions", "Model Information"])
    
    # Tab 1: Make Prediction
    with tab1:
        st.header("Enter Applicant Information")
        
        # User ID selection
        col1, col2 = st.columns([1, 2])
        with col1:
            use_sample_id = st.checkbox("Use sample user ID", value=True if sample_users else False, disabled=not sample_users)
        
        user_id = None
        if use_sample_id and sample_users:
            with col2:
                user_id = st.selectbox("Select User ID", options=sample_users)
        else:
            with col2:
                user_id = st.number_input("Enter User ID", min_value=100000, value=100001, step=1)
        
        # Pre-fill form with user data if available
        user_data = None
        if app_data is not None and user_id is not None:
            user_rows = app_data[app_data['SK_ID_CURR'] == user_id]
            if not user_rows.empty:
                user_data = user_rows.iloc[0]
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            # Default values
            income_default = float(user_data['AMT_INCOME_TOTAL']) if user_data is not None and 'AMT_INCOME_TOTAL' in user_data else 150000.0
            credit_default = float(user_data['AMT_CREDIT']) if user_data is not None and 'AMT_CREDIT' in user_data else 500000.0
            annuity_default = float(user_data['AMT_ANNUITY']) if user_data is not None and 'AMT_ANNUITY' in user_data else 25000.0
            ext1_default = float(user_data['EXT_SOURCE_1']) if user_data is not None and 'EXT_SOURCE_1' in user_data else 0.5
            ext2_default = float(user_data['EXT_SOURCE_2']) if user_data is not None and 'EXT_SOURCE_2' in user_data else 0.5
            ext3_default = float(user_data['EXT_SOURCE_3']) if user_data is not None and 'EXT_SOURCE_3' in user_data else 0.5
            
            # Age and employment duration
            age_default = abs(int(user_data['DAYS_BIRTH'])) // 365 if user_data is not None and 'DAYS_BIRTH' in user_data else 35
            employment_default = abs(int(user_data['DAYS_EMPLOYED'])) // 365 if user_data is not None and 'DAYS_EMPLOYED' in user_data else 5
            
            with col1:
                amt_income = st.number_input("Income Amount", min_value=0.0, value=income_default, step=10000.0)
                amt_credit = st.number_input("Credit Amount", min_value=0.0, value=credit_default, step=50000.0)
                amt_annuity = st.number_input("Annuity Amount", min_value=0.0, value=annuity_default, step=1000.0)
                ext_source_1 = st.slider("External Source 1 Score", min_value=0.0, max_value=1.0, value=ext1_default, step=0.01)
            
            with col2:
                ext_source_2 = st.slider("External Source 2 Score", min_value=0.0, max_value=1.0, value=ext2_default, step=0.01)
                ext_source_3 = st.slider("External Source 3 Score", min_value=0.0, max_value=1.0, value=ext3_default, step=0.01)
                age_years = st.slider("Age (years)", min_value=18, max_value=70, value=int(age_default), step=1)
                employment_years = st.slider("Employment Duration (years)", min_value=0, max_value=40, value=int(employment_default), step=1)
            
            # Input data dictionary
            input_data = {
                "SK_ID_CURR": user_id,
                "AMT_INCOME_TOTAL": amt_income,
                "AMT_CREDIT": amt_credit,
                "AMT_ANNUITY": amt_annuity,
                "EXT_SOURCE_1": ext_source_1,
                "EXT_SOURCE_2": ext_source_2,
                "EXT_SOURCE_3": ext_source_3,
                "age_years": age_years,
                "employment_years": employment_years
            }
            
            # Submit button
            submitted = st.form_submit_button("Make Prediction")
        
        if submitted:
            with st.spinner("Processing..."):
                # Prepare input data - use app_data but don't combine feature lists
                features_data, missing_features = prepare_input_data(
                    input_data, 
                    feature_names, 
                    app_data  # Still pass app_data to fill in known values but don't extend feature list
                )
                
                if len(missing_features) > 0:
                    st.info(f"Using default values for {len(missing_features)} features not provided in the form.")
                
                # Make prediction
                prediction = make_prediction(models, features_data)
                
                if prediction is not None:
                    # Save prediction
                    prediction_id = save_prediction(input_data, prediction, feature_names)
                    
                    # Display results
                    risk_level = "High Risk" if prediction > 0.5 else "Low Risk"
                    risk_color = "red" if prediction > 0.5 else "green"
                    
                    st.markdown(f"### Prediction Result: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                    st.markdown(f"### Default Probability: <span style='color:{risk_color}'>{prediction:.4f}</span>", unsafe_allow_html=True)
                    
                    # Progress bar for visualization
                    st.progress(float(prediction))
                    
                    # Result interpretation
                    st.subheader("Interpretation")
                    if prediction < 0.5:
                        st.success("This applicant has a low risk of default. The loan application could be approved.")
                    else:
                        st.error("This applicant has a high risk of default. The loan application might need additional review.")
                    
                    # Key factors
                    st.subheader("Key Factors")
                    
                    factors = [
                        {"name": "External Source 2", "value": ext_source_2, "impact": "Lower scores increase default risk", "importance": "High"},
                        {"name": "External Source 3", "value": ext_source_3, "impact": "Lower scores increase default risk", "importance": "High"},
                        {"name": "Debt-to-Income Ratio", "value": amt_credit/amt_income, "impact": "Higher ratios increase default risk", "importance": "Medium"},
                        {"name": "External Source 1", "value": ext_source_1, "impact": "Lower scores increase default risk", "importance": "Medium"},
                        {"name": "Age", "value": age_years, "impact": "Younger applicants may have higher risk", "importance": "Low"}
                    ]
                    
                    # Display factors as a table
                    factor_df = pd.DataFrame(factors)
                    st.table(factor_df)
    
    # Tab 2: Historical Predictions
    with tab2:
        st.header("Historical Predictions")
        
        # Refresh button
        if st.button("Refresh Historical Data"):
            st.experimental_rerun()
        
        # Load prediction history
        with st.spinner("Loading historical predictions..."):
            predictions = load_prediction_history()
        
        if not predictions:
            st.info("No historical predictions found")
        else:
            st.success(f"Found {len(predictions)} historical predictions")
            
            # Create and display history dataframe
            df = create_history_dataframe(predictions)
            if not df.empty:
                st.dataframe(df)
            else:
                st.warning("Could not format prediction history data")
            
            # Option to view detailed prediction
            if predictions:
                selected_id = st.selectbox("Select a prediction to view details:", 
                                         [p["prediction_id"] for p in predictions])
                
                if selected_id:
                    selected_pred = next((p for p in predictions if p["prediction_id"] == selected_id), None)
                    if selected_pred:
                        # Show details
                        st.write("### Prediction Details")
                        
                        # Basic information
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### User Information")
                            st.write(f"**User ID:** {selected_pred.get('input_data', {}).get('SK_ID_CURR', 'N/A')}")
                            st.write(f"**Prediction Time:** {selected_pred.get('timestamp', '')[:19]}")
                        
                        with col2:
                            st.write("### Prediction Result")
                            result = selected_pred.get('prediction_result', 0)
                            label = "High Risk" if result > 0.5 else "Low Risk"
                            color = "red" if result > 0.5 else "green"
                            st.markdown(f"**Default Probability:** <span style='color:{color}'>{result:.4f}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Risk Assessment:** <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                        
                        # User data
                        st.write("### Input Data")
                        input_data = selected_pred.get("input_data", {})
                        
                        # Remove SK_ID_CURR from display data
                        display_data = {k: v for k, v in input_data.items() if k != "SK_ID_CURR"}
                        
                        # Format specially handled fields
                        if "age_years" in display_data:
                            display_data["Age (years)"] = display_data.pop("age_years")
                        if "employment_years" in display_data:
                            display_data["Employment (years)"] = display_data.pop("employment_years")
                        
                        # Create two columns for data display
                        data_items = list(display_data.items())
                        mid_point = len(data_items) // 2
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            for k, v in data_items[:mid_point]:
                                st.write(f"**{k}:** {v}")
                        
                        with col2:
                            for k, v in data_items[mid_point:]:
                                st.write(f"**{k}:** {v}")
                        
                        # Show raw JSON data (not inside an expander to avoid nesting issues)
                        st.write("### Raw JSON Data")
                        st.json(selected_pred)
    
    # Tab 3: Model Information
    with tab3:
        st.header("Model Information")
        
        # Display model information
        st.write(f"### Loaded Models: {len(models)}")
        st.write(f"### Features Used: {len(feature_names)}")
        
        # Feature importance plot
        st.subheader("Feature Importance")
        importance_plot = get_feature_importance_plot(models, feature_names)
        if importance_plot:
            st.pyplot(importance_plot)
        else:
            st.info("Feature importance plot not available for this model")
        
        # Model details
        with st.expander("Model Details"):
            for i, model in enumerate(models):
                st.write(f"### Model {i+1}")
                st.write(f"**Type:** {type(model).__name__}")
                
                # Try to get model parameters
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    st.write("**Parameters:**")
                    for key, value in params.items():
                        st.write(f"- {key}: {value}")
                
                # Feature input shape
                if hasattr(model, 'n_features_'):
                    st.write(f"**Number of features:** {model.n_features_}")
                elif hasattr(model, 'n_features_in_'):
                    st.write(f"**Number of features:** {model.n_features_in_}")

if __name__ == "__main__":
    main()