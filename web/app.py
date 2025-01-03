"""Inference Server.

   Run application with:

   $ flask run
"""

import dtreeviz
import joblib
import matplotlib
import pandas as pd
from dtreeviz.utils import extract_params_from_pipeline
from flask import Flask, jsonify, render_template, request

# Use a non-GUI backend
matplotlib.use("Agg")

# PRETTY_FEATURE_NAMES = {
#     "num__age" : "Age",
#     "num__size" : "Size",
#     "num__ri" : "RI",
#     "pas__palpable" : "Palpable",
#     "pas__vessels" : "Vessels",
#     "cat__shape_oval" : "Shape_oval",
#     "cat__shape_round" : "Shape_round",
#     "cat__shape_irregular" : "Shape_irregular",
#     "cat__margins_circumscribed" : "Margins_circumscribed",
#     "cat__margins_indistinct" : "Margins_indistinct",
#     "cat__margins_angular" : "Margins_angular",
#     "cat__margins_microlobulated" : "Margins_microlobulated",
#     "cat__margins_spiculated" : "Margins_spiculated",
#     "cat__orientation_parallel" : "Orientation_parallel",
#     "cat__orientation_not parallel" : "Orientation_not_parallel",
# }

# Load pre-trained stuff
with open("lr.joblib", "rb") as model_file:
    best_lr_model = joblib.load(model_file)
with open("dt.joblib", "rb") as model_file:
    best_tree_model = joblib.load(model_file)
with open("residual_df.joblib", "rb") as dt_train_data:
    residual_df = joblib.load(dt_train_data)
    X_new, y_new = residual_df.drop(columns=["residuals"]), residual_df.residuals
with open("quantiles_by_leaf.joblib", "rb") as model_file:
    leaf2quantile = joblib.load(model_file)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    d = request.form.to_dict()
    x = pd.DataFrame(data=[d.values()], columns=d.keys())
    x_lr_processed = best_lr_model["col_transf"].transform(x).astype(float)
    x_dt_processed = best_tree_model["col_transf"].transform(x).astype(float)
    print(f"\nIncoming data:\n{x}\n")

    # Predict risk of malignancy
    prediction = best_lr_model.predict_proba(x)[0]
    print(f"Risk of Cancer: {100*prediction[1]:.2f}%\n")

    # Display the lr coeffs with feature names
    print("Logistic Regression Coefficients:")
    lr_coefficients = best_lr_model["lr"].coef_.flatten()
    feature_names = best_lr_model["col_transf"].get_feature_names_out()
    for feature, coef in dict(zip(feature_names, lr_coefficients)).items():
        print(f"{feature}: {coef:.3f}")

    # Compute top 3 features contributing to this prediction
    print("\nFeature Impact:")
    impact_values = lr_coefficients * x_lr_processed.flatten()
    feature_impact = list(zip(feature_names, impact_values))
    for feature, impact in feature_impact:
        print(f"{feature}: {impact:.3f}")
    # top_3_features = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)[:3]
    top_3_features = sorted(feature_impact, key=lambda x: x[1], reverse=True)[:3]
    print(f"\nTop 3:\n{top_3_features}")

    # Quantifying uncertainty
    leaf = best_tree_model["dt"].apply(x_dt_processed)[0]
    quantile = leaf2quantile[leaf]
    prediction_set = dict()
    for i, p in enumerate(prediction):
        if p >= quantile:
            prediction_set[i] = p
    print(f"\nPrediction set: {prediction_set}")

    # Visualizing decision path
    model, X_dtreeviz, features = extract_params_from_pipeline(
        pipeline=best_tree_model, X_train=X_new, feature_names=X_new.columns  # FIXME: cat__* and pas__* are too long!
    )
    viz_model = dtreeviz.model(
        model=model, X_train=X_dtreeviz, y_train=y_new, feature_names=features, target_name="residuals"
    )
    print(x_dt_processed)
    print(model, X_dtreeviz, features)
    decision_path = viz_model.view(
        orientation="TD",
        show_just_path=True,
        x=x_dt_processed[0],
        fancy=True,
        histtype="barstacked",
        leaftype="pie",
        fontname="monospace",
        # colors={"scatter_marker": "pink"},
    )

    try:
        return jsonify(
            result=f"{prediction[1]*100:.2f}%",
            p0="{prediction[0]*100:.2f}%",
            p1="{prediction[1]*100:.2f}%",
            top_3_features=top_3_features,
            uncertainty_set=prediction_set,
            quantile_threshold=leaf2quantile[leaf],
            decision_path=None,
            svg_plot=decision_path.svg(),
        )
    except ValueError as e:
        return jsonify(result=f"Error: {str(e)}"), 400


if __name__ == "__main__":
    app.run(debug=True)
