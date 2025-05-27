import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import gc
import psutil

def load_pickle(filename: str):
    """Load pickle with memory monitoring"""
    print(f"📁 Loading {filename}...")
    
    # Check available memory before loading
    memory = psutil.virtual_memory()
    print(f"💾 Available memory: {memory.available / (1024**3):.2f} GB")
    
    with open(filename, "rb") as f_in:
        data = pickle.load(f_in)
    
    print(f"✅ Loaded {filename} successfully")
    return data

def run_train(data_path: str):
    try:
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        print(f"🔍 Initial memory usage: {process.memory_info().rss / (1024**2):.2f} MB")
        
        # Load data in smaller chunks if possible
        print("📊 Loading training data...")
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        
        print("📊 Loading validation data...")
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        
        print(f"🔍 Memory after loading: {process.memory_info().rss / (1024**2):.2f} MB")
        print(f"📏 Training set shape: {X_train.shape}")
        print(f"📏 Validation set shape: {X_val.shape}")
        
        # Use more memory-efficient RandomForest parameters
        rf = RandomForestRegressor(
            n_estimators=50,      # Reduced from default 100
            max_depth=10,         # Limit tree depth
            max_features='sqrt',  # Reduce feature sampling
            n_jobs=1,            # Single thread to control memory
            random_state=0
        )
        
        print("🎯 Training model...")
        rf.fit(X_train, y_train)
        
        print("🔮 Making predictions...")
        y_pred = rf.predict(X_val)
        
        # Calculate metrics
        rmse = sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)
        
        # Log model parameters
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("max_features", "sqrt")
        mlflow.log_param("min_samples_split", rf.min_samples_split)
        
        print(f"✅ RMSE: {rmse:.4f}")
        
        # Clean up memory
        del X_train, X_val, y_train, y_val, y_pred
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

if __name__ == '__main__':
    print("🚀 Starting training...")
    
    try:
        # Use local MLflow instead of remote to avoid connection issues
        mlflow.set_tracking_uri("https://mlflow-q10f8q.adv-cml01.apps.advocprdc.stc.com.sa")
        
        # Create experiment locally
        experiment_name = "my-experiment"
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass  # Experiment already exists
            
        mlflow.set_experiment(experiment_name)
        
        # Disable autolog to reduce memory overhead
        # mlflow.autolog()  # Comment this out to reduce memory usage
        
        with mlflow.start_run():
            run_train("/home/cdsw/my_codes/output")
            
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)