"""Employee Attrition Intelligence Platform - Streamlit entry point."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from app.components.ui import metric_row, page_title, risk_badge, safe_plotly_bar
from app.utils.artifacts import (
    load_calibrated_model,
    load_csv_report,
    load_employee_data,
    load_json_report,
    load_risk_config,
    load_tuned_logistic_model,
    model_feature_columns,
)
from app.utils.paths import project_path
from app.utils.prediction import build_profile_frame, predict_profile
from src.utils.logging_config import configure_logging


configure_logging().info("Streamlit dashboard startup.")


st.set_page_config(
    page_title="Employee Attrition Intelligence Platform",
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_historical_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate department-level historical employee and attrition metrics."""
    return (
        frame.groupby("Department", observed=True)
        .agg(
            employee_count=("Attrition", "size"),
            historical_attrition_count=("Attrition", "sum"),
            historical_attrition_rate=("Attrition", "mean"),
        )
        .reset_index()
    )


def render_executive() -> None:
    page_title(
        "Executive Overview",
        "Historical workforce outcomes and model-supported attrition risk analytics.",
    )
    try:
        frame = load_employee_data()
        historical = load_historical_rates(frame)
        risk_config = load_risk_config()
        risk_scores = pd.read_csv(
            project_path("data", "processed", "employee_risk_scores.csv")
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Required artifact could not be loaded: {exc}")
        return

    attrition_count = int(frame["Attrition"].sum())
    attrition_rate = attrition_count / len(frame)
    metric_row(
        [
            ("Total Employees", f"{len(frame):,}"),
            ("Historical Attrition", f"{attrition_count:,}"),
            ("Historical Attrition Rate", f"{attrition_rate:.2%}"),
            ("Held-Out Test Records", "294"),
        ]
    )

    left, right = st.columns(2)
    with left:
        department = (
            frame.groupby("Department", observed=True)
            .size()
            .reset_index(name="Employee Count")
        )
        figure = px.bar(
            department,
            x="Department",
            y="Employee Count",
            title="Department Employee Distribution",
            template="plotly_white",
        )
        st.plotly_chart(figure, width="stretch")

    with right:
        historical["Historical Attrition Rate"] = (
            historical["historical_attrition_rate"] * 100
        )
        figure = px.bar(
            historical,
            x="Department",
            y="Historical Attrition Rate",
            title="Historical Attrition Rate by Department",
            template="plotly_white",
        )
        figure.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(figure, width="stretch")

    if "risk_category" in risk_scores.columns:
        risk_distribution = (
            risk_scores["risk_category"]
            .value_counts()
            .rename_axis("Risk Category")
            .reset_index(name="Employee Records")
        )
        figure = px.pie(
            risk_distribution,
            names="Risk Category",
            values="Employee Records",
            title="Held-Out Predicted Risk Distribution",
            color="Risk Category",
            color_discrete_map={
                "LOW": "#2ca02c",
                "MEDIUM": "#ff7f0e",
                "HIGH": "#d62728",
            },
        )
        st.plotly_chart(figure, width="stretch")

    st.subheader("Key Business Insights")
    overtime = (
        frame.groupby("OverTime")["Attrition"].mean().mul(100).to_dict()
    )
    insight_columns = st.columns(3)
    insight_columns[0].info(
        f"Historical overtime attrition: "
        f"Yes {overtime.get('Yes', 0):.2f}% vs No "
        f"{overtime.get('No', 0):.2f}%."
    )
    insight_columns[1].info(
        "Risk scores describe historical-pattern propensity, "
        "not a guaranteed resignation within 30, 60, or 90 days."
    )
    insight_columns[2].warning(
        "Sensitive demographic fields are excluded from operational predictions."
    )


def form_profile(frame: pd.DataFrame, prefix: str) -> tuple[dict, pd.DataFrame]:
    """Render an operational hypothetical employee form."""
    model_columns, _ = model_feature_columns()
    st.subheader("Supported operational attributes")

    left, right = st.columns(2)
    values = {}
    with left:
        if "Department" in model_columns:
            values["Department"] = st.selectbox(
                "Department",
                sorted(frame["Department"].dropna().unique()),
                key=f"{prefix}_department",
            )
        if "JobRole" in model_columns:
            values["JobRole"] = st.selectbox(
                "Job Role",
                sorted(frame["JobRole"].dropna().unique()),
                key=f"{prefix}_job_role",
            )
        if "OverTime" in model_columns:
            values["OverTime"] = st.selectbox(
                "Overtime",
                ["No", "Yes"],
                key=f"{prefix}_overtime",
            )
        if "MonthlyIncome" in model_columns:
            values["MonthlyIncome"] = st.number_input(
                "Monthly Income",
                min_value=0,
                max_value=int(frame["MonthlyIncome"].max() * 2),
                value=int(frame["MonthlyIncome"].median()),
                step=500,
                key=f"{prefix}_monthly_income",
            )
        if "JobSatisfaction" in model_columns:
            values["JobSatisfaction"] = st.slider(
                "Job Satisfaction",
                1,
                4,
                int(frame["JobSatisfaction"].median()),
                key=f"{prefix}_job_satisfaction",
            )
        if "WorkLifeBalance" in model_columns:
            values["WorkLifeBalance"] = st.slider(
                "Work-Life Balance",
                1,
                4,
                int(frame["WorkLifeBalance"].median()),
                key=f"{prefix}_work_life",
            )

    with right:
        if "YearsAtCompany" in model_columns:
            values["YearsAtCompany"] = st.slider(
                "Years at Company",
                0,
                int(frame["YearsAtCompany"].max()),
                int(frame["YearsAtCompany"].median()),
                key=f"{prefix}_years_company",
            )
        if "TotalWorkingYears" in model_columns:
            values["TotalWorkingYears"] = st.slider(
                "Total Working Years",
                0,
                int(frame["TotalWorkingYears"].max()),
                int(frame["TotalWorkingYears"].median()),
                key=f"{prefix}_total_years",
            )
        if "YearsInCurrentRole" in model_columns:
            values["YearsInCurrentRole"] = st.slider(
                "Years in Current Role",
                0,
                int(frame["YearsInCurrentRole"].max()),
                int(frame["YearsInCurrentRole"].median()),
                key=f"{prefix}_years_role",
            )
        if "DistanceFromHome" in model_columns:
            values["DistanceFromHome"] = st.slider(
                "Distance From Home",
                0,
                int(frame["DistanceFromHome"].max()),
                int(frame["DistanceFromHome"].median()),
                key=f"{prefix}_distance",
            )

    profile = build_profile_frame(frame, model_columns, values)
    return values, profile


def underlying_shap_explanation(profile: pd.DataFrame) -> list[dict]:
    """Generate underlying-logistic SHAP contributions for a hypothetical profile."""
    try:
        tuned = load_tuned_logistic_model()
        split = joblib.load(project_path("data", "processed", "phase3_split.joblib"))
        frame = load_employee_data()
        model_columns, _ = model_feature_columns()
        train_indices = split["train_indices"]
        train = frame.loc[train_indices, model_columns]

        preprocessing = tuned.named_steps["preprocessing"]
        transformed_train = preprocessing.transform(train)
        if hasattr(transformed_train, "toarray"):
            transformed_train = transformed_train.toarray()
        transformed_profile = preprocessing.transform(profile)
        if hasattr(transformed_profile, "toarray"):
            transformed_profile = transformed_profile.toarray()

        background = transformed_train[:100]
        classifier = tuned.named_steps["classifier"]
        explainer = shap.LinearExplainer(classifier, background)
        shap_values = explainer.shap_values(transformed_profile)
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        names = preprocessing.named_steps["preprocessor"].get_feature_names_out()
        names = [str(name).replace("_", " ", 1) for name in names]
        vector = shap_values[0]
        ranking = np.argsort(np.abs(vector))[::-1][:8]
        return [
            {
                "feature": names[index],
                "shap_value": float(vector[index]),
            }
            for index in ranking
        ]
    except (FileNotFoundError, OSError, ValueError, KeyError, AttributeError):
        return []


def render_risk_assessment() -> None:
    page_title(
        "Employee Risk Assessment",
        "Hypothetical employee-level historical attrition propensity.",
    )
    try:
        frame = load_employee_data()
        model = load_calibrated_model()
        risk_config = load_risk_config()
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Required prediction artifact could not be loaded: {exc}")
        return

    _, profile = form_profile(frame, "assessment")
    threshold = float(risk_config["threshold"])
    if st.button("Calculate Attrition Propensity", type="primary"):
        try:
            result = predict_profile(model, profile, threshold)
        except (ValueError, TypeError) as exc:
            st.error(f"Prediction could not be calculated: {exc}")
            return

        left, right = st.columns([1, 2])
        with left:
            st.metric(
                "Calibrated Attrition Propensity",
                f"{result.probability:.1%}",
            )
            risk_badge(result.risk_category)
            st.caption(
                f"0.21 classification threshold prediction: "
                f"{'Positive' if result.threshold_prediction else 'Negative'}"
            )
        with right:
            st.progress(min(result.probability, 1.0))
            st.markdown(
                "This is a historical-pattern-based propensity estimate. "
                "It is not a guarantee or a 30/60/90-day resignation forecast."
            )

        contributors = underlying_shap_explanation(profile)
        if contributors:
            st.subheader("Underlying Logistic Regression Contributions")
            contribution = pd.DataFrame(contributors)
            figure = px.bar(
                contribution.sort_values("shap_value"),
                x="shap_value",
                y="feature",
                orientation="h",
                title="Top SHAP Contributions",
                template="plotly_white",
            )
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "SHAP explains the underlying linear model score, not the exact "
                "isotonic-calibrated probability."
            )


def render_department_analytics() -> None:
    page_title("Department Analytics", "Department workforce and risk comparisons.")
    try:
        frame = load_employee_data()
        department = (
            frame.groupby("Department", observed=True)
            .agg(
                employee_count=("Attrition", "size"),
                historical_attrition_count=("Attrition", "sum"),
                historical_attrition_rate=("Attrition", "mean"),
            )
            .reset_index()
        )
        department["historical_attrition_rate"] *= 100
        held_out = pd.read_csv(
            project_path("data", "processed", "department_risk_summary.csv")
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Department artifacts could not be loaded: {exc}")
        return

    selected = st.multiselect(
        "Filter Departments",
        sorted(department["Department"].unique()),
        default=sorted(department["Department"].unique()),
    )
    department = department[department["Department"].isin(selected)]
    held_out = held_out[held_out["department"].isin(selected)]

    left, right = st.columns(2)
    with left:
        safe_plotly_bar(
            department,
            "Department",
            "employee_count",
            "Employee Count",
        )
    with right:
        safe_plotly_bar(
            department,
            "Department",
            "historical_attrition_rate",
            "Historical Attrition Rate (%)",
        )
    if not held_out.empty:
        safe_plotly_bar(
            held_out,
            "department",
            "mean_attrition_probability",
            "Held-Out Mean Predicted Propensity",
        )


def render_workforce_insights() -> None:
    page_title("Workforce Insights", "Exploratory patterns from the supplied dataset.")
    try:
        frame = load_employee_data()
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Dataset could not be loaded: {exc}")
        return

    left, right = st.columns(2)
    with left:
        overtime = (
            frame.groupby("OverTime")["Attrition"].mean().mul(100).reset_index()
        )
        overtime.rename(
            columns={"Attrition": "Historical Attrition Rate"},
            inplace=True,
        )
        safe_plotly_bar(
            overtime,
            "OverTime",
            "Historical Attrition Rate",
            "Overtime and Historical Attrition",
        )
    with right:
        job_role = (
            frame.groupby("JobRole")["Attrition"].mean().mul(100).reset_index()
        )
        job_role.rename(
            columns={"Attrition": "Historical Attrition Rate"},
            inplace=True,
        )
        safe_plotly_bar(
            job_role,
            "JobRole",
            "Historical Attrition Rate",
            "Job Role and Historical Attrition",
        )

    st.plotly_chart(
        px.box(
            frame,
            x="Attrition",
            y="YearsAtCompany",
            title="Tenure and Historical Attrition",
            template="plotly_white",
        ),
        width="stretch",
    )
    st.caption(
        "Charts show historical associations and do not establish causal retention effects."
    )


def render_explainability() -> None:
    page_title("Explainability Center", "Global and individual model explanations.")
    try:
        importance = load_csv_report(
            "reports/explainability/shap_global_importance.csv"
        )
        coefficients = load_csv_report(
            "reports/explainability/logistic_coefficients.csv"
        )
        individuals = load_json_report(
            "reports/explainability/individual_explanations.json"
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Explainability artifacts could not be loaded: {exc}")
        return

    st.warning(
        "SHAP explains the underlying Logistic Regression score. "
        "It does not exactly explain the final isotonic-calibrated probability."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Global SHAP Importance")
        summary_path = project_path(
            "reports",
            "figures",
            "phase7",
            "phase7_shap_summary.png",
        )
        if summary_path.exists():
            st.image(str(summary_path), width="stretch")
    with right:
        st.subheader("SHAP Bar Chart")
        bar_path = project_path(
            "reports",
            "figures",
            "phase7",
            "phase7_shap_feature_importance.png",
        )
        if bar_path.exists():
            st.image(str(bar_path), width="stretch")

    st.subheader("Top SHAP Features")
    st.dataframe(
        importance.head(20),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Coefficient Comparison")
    st.dataframe(
        coefficients.head(20),
        width="stretch",
        hide_index=True,
    )

    if individuals:
        st.subheader("Existing Individual Explanation")
        record_ids = [item["anonymous_record_id"] for item in individuals]
        selected = st.selectbox("Select explanation", record_ids)
        item = next(
            candidate
            for candidate in individuals
            if candidate["anonymous_record_id"] == selected
        )
        st.metric(
            "Calibrated Propensity",
            f"{item['calibrated_attrition_propensity']:.1%}",
        )
        risk_badge(item["risk_category"])
        st.write(item["human_readable_explanation"])


def render_fairness() -> None:
    page_title("Fairness & Responsible AI", "Held-out demographic subgroup audit.")
    try:
        fairness = load_csv_report("reports/DEMOGRAPHIC_METRICS.csv")
        metadata = load_json_report("reports/phase8_fairness_metadata.json")
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Fairness artifacts could not be loaded: {exc}")
        return

    audit = st.selectbox(
        "Demographic audit",
        sorted(fairness["audit"].unique()),
    )
    table = fairness[fairness["audit"] == audit]
    st.dataframe(
        table[
            [
                "group",
                "sample_size",
                "selection_rate",
                "tpr_recall",
                "fpr",
                "fnr",
                "precision",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    small = table[table["sample_size"] < 30]
    if not small.empty:
        st.warning(
            "Small subgroup warning: "
            + ", ".join(small["group"].astype(str).tolist())
            + " have fewer than 30 held-out records."
        )

    for metric, title in [
        ("selection_rate", "Selection/Prediction Rate"),
        ("tpr_recall", "Recall / True-Positive Rate"),
        ("fpr", "False-Positive Rate"),
    ]:
        figure = px.bar(
            table,
            x="group",
            y=metric,
            title=f"{audit}: {title}",
            template="plotly_white",
        )
        st.plotly_chart(figure, width="stretch")

    st.subheader("Responsible AI Guidelines")
    guideline_path = project_path(
        "reports",
        "RESPONSIBLE_AI_GUIDELINES.md",
    )
    if guideline_path.exists():
        st.markdown(guideline_path.read_text(encoding="utf-8"))
    else:
        st.info("Responsible AI guidelines artifact is unavailable.")


def render_scenarios() -> None:
    page_title("What-If Scenario Lab", "Hypothetical model comparisons.")
    try:
        frame = load_employee_data()
        model = load_calibrated_model()
        risk_config = load_risk_config()
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Scenario artifacts could not be loaded: {exc}")
        return

    st.info(
        "Scenario changes are hypothetical model outputs. "
        "They do not demonstrate causal effects or guaranteed outcomes."
    )
    _, baseline_profile = form_profile(frame, "baseline")
    st.divider()
    st.subheader("Scenario Changes")
    _, scenario_profile = form_profile(frame, "scenario")

    threshold = float(risk_config["threshold"])
    baseline = predict_profile(model, baseline_profile, threshold)
    scenario = predict_profile(model, scenario_profile, threshold)

    left, right = st.columns(2)
    left.metric(
        "Baseline Propensity",
        f"{baseline.probability:.1%}",
        delta=f"{baseline.risk_category} risk",
    )
    right.metric(
        "Scenario Propensity",
        f"{scenario.probability:.1%}",
        delta=f"{scenario.probability - baseline.probability:+.1%}",
    )
    st.caption(
        "No intervention is claimed to cause the displayed probability change."
    )


def render_model_performance() -> None:
    page_title("Model Performance", "Actual held-out test validation and calibration.")
    try:
        metadata = load_json_report("reports/phase6_validation_metadata.json")
        test_validation = load_csv_report("reports/test_model_validation.csv")
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Model performance artifacts could not be loaded: {exc}")
        return

    held_out = metadata["held_out_metrics"]
    metric_row(
        [
            ("PR-AUC", f"{held_out['pr_auc']:.4f}"),
            ("Recall", f"{held_out['recall']:.4f}"),
            ("F1-Score", f"{held_out['f1']:.4f}"),
            ("Test Records", f"{metadata['test_row_count']:,}"),
        ]
    )
    st.caption(
        f"Classification threshold: {metadata['selected_threshold']:.2f}; "
        "Low/Medium/High boundaries are separate propensity categories."
    )

    st.subheader("Candidate Held-Out Validation")
    st.dataframe(
        test_validation,
        width="stretch",
        hide_index=True,
    )

    roc_path = project_path(
        "reports",
        "figures",
        "phase6",
        "phase6_test_roc.png",
    )
    pr_path = project_path(
        "reports",
        "figures",
        "phase6",
        "phase6_test_precision_recall.png",
    )
    confusion_path = project_path(
        "reports",
        "figures",
        "phase6",
        "phase6_test_confusion_matrix.png",
    )
    left, right = st.columns(2)
    if roc_path.exists():
        left.image(str(roc_path), width="stretch")
    if pr_path.exists():
        right.image(str(pr_path), width="stretch")
    if confusion_path.exists():
        st.image(str(confusion_path), width="stretch")

    st.subheader("Model and Calibration")
    st.write(
        f"- Selected model: {metadata['selected_final_model']}\n"
        f"- Calibration: {metadata['selected_calibration_method']}\n"
        f"- Calibrated test Brier score: "
        f"{metadata['brier_score_calibrated_test']:.4f}"
    )


pages = {
    "Executive Overview": render_executive,
    "Employee Risk Assessment": render_risk_assessment,
    "Department Analytics": render_department_analytics,
    "Workforce Insights": render_workforce_insights,
    "Explainability Center": render_explainability,
    "Fairness & Responsible AI": render_fairness,
    "What-If Scenario Lab": render_scenarios,
    "Model Performance": render_model_performance,
}

st.sidebar.title("HR Analytics")
st.sidebar.caption("Employee Attrition Intelligence Platform")
selected = st.sidebar.radio("Navigation", list(pages))
st.sidebar.divider()
st.sidebar.caption(
    "Historical-pattern analytics for human-reviewed retention research."
)

pages[selected]()
