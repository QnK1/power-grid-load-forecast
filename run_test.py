# run_test.py

import numpy as np
from models.committee_system import CommitteeSystem
from models.dummy_model import DummyModel

def main():
    """
    Main function to test the CommitteeSystem with dummy models.
    """
    print("--- Starting Committee System Test ---")

    # 1. Create the 'expert' models for our committee
    # These are our placeholder "players"
    print("\nStep 1: Initializing dummy models...")
    dummy_mlp = DummyModel(name="Dummy MLP")
    dummy_lstm = DummyModel(name="Dummy LSTM")
    dummy_cnn = DummyModel(name="Dummy CNN")
    
    expert_models = [dummy_mlp, dummy_lstm, dummy_cnn]

    # 2. Create the CommitteeSystem itself
    # This is our "manager"
    print("\nStep 2: Initializing the Committee System...")
    committee = CommitteeSystem(models=expert_models)

    # 3. Create some fake data to test with
    # In the future, this will come from our real dataset
    print("\nStep 3: Generating fake data for testing...")
    # 10 samples, 5 features each (the numbers don't matter for this test)
    X_fake_train = np.random.rand(100, 5)
    y_fake_train = np.random.rand(100, 1)
    X_fake_test = np.random.rand(10, 5) 
    print("Fake data created.")

    # 4. "Train" the committee
    print("\nStep 4: Calling the train method...")
    committee.train(X_fake_train, y_fake_train)

    # 5. Make a "prediction"
    print("\nStep 5: Calling the predict method...")
    final_predictions = committee.predict(X_fake_test)
    print("Prediction complete.")

    # 6. Print the results to verify it worked
    print("\n--- Test Results ---")
    print(f"Shape of the final predictions: {final_predictions.shape}")
    print("First 5 predictions:")
    print(final_predictions[:5])
    print("\n--- Test Finished Successfully ---")


if __name__ == "__main__":
    main()