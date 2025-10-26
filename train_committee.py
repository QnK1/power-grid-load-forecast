import pandas as pd
from models.committee_modular_system import CommitteeSystem
from models.modular_system.modular_system_wrapper import ModularSystemWrapper
from utils.load_data import load_training_data, load_test_data, DataLoadingParams, decode_ml_outputs

# --- FIX: Use the same list of 9 features for consistency ---
FEATURE_COLUMNS = [
    'load_timestamp_-1', 'load_timestamp_-2', 'load_timestamp_-3',
    'prev_3_temperature_timestamps_mean',
    'prev_day_temperature_5_timestamps_mean',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos'
]

def main():
    """Main script to orchestrate the full pipeline."""
    print("--- Starting Full Training & Evaluation Pipeline ---")

    print("\nStep 1: Initializing models...")
    modular_model = ModularSystemWrapper(hidden_layers=[[24, 12]], epochs=[20])
    committee = CommitteeSystem(models=[modular_model])

    print("\nStep 2: Training the committee...")
    # This calls the teammate's script, which does its own data loading
    committee.train(X_train=None, y_train=None)

    print("\nStep 3: Loading data for evaluation...")
    params = DataLoadingParams()
    params.shuffle = False
    params.prev_day_load_values = (0, 0) # Ensure we get the same features
    _, y_train_raw = load_training_data(params)
    X_test_raw, y_test_raw = load_test_data(params)
    
    # Ensure the test data has the correct 9 features for prediction
    X_test = X_test_raw[FEATURE_COLUMNS]

    print("\nStep 4: Making predictions...")
    scaled_predictions = committee.predict(X_test)

    print("\nStep 5: Decoding predictions to real values...")
    final_predictions = decode_ml_outputs(scaled_predictions, y_train_raw)

    print("\n--- Final Results ---")
    results_df = pd.DataFrame({
        'Actual Load (MW)': y_test_raw['load'].values,
        'Predicted Load (MW)': final_predictions.flatten()
    }, index=y_test_raw.index)
    
    print("Comparison of Actual vs. Predicted values (first 24 hours):")
    print(results_df.head(24))

if __name__ == "__main__":
    main()