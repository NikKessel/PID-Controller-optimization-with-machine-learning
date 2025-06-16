import joblib

# Adjust path as needed
model = joblib.load(r"D:\BA\PID-Controller-optimization-with-machine-learning\src\streamlit\streamlit_models\model_surrogate.joblib")
print(model.feature_names_in_)
