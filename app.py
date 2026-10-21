from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Route Resilience",
    page_icon="🛣️",
    layout="wide"
)


OUTPUTS = Path("outputs")


def output_file(filename):
    return OUTPUTS / filename


st.title("Route Resilience")
st.caption(
    "Occlusion-robust road extraction and graph-theoretic "
    "criticality analysis"
)

st.info(
    "Prototype pipeline: satellite image → road prediction → "
    "gap repair → road graph → resilience simulation."
)


st.header("Pipeline Outputs")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted Road Overlay")

    path = output_file(
        "predicted_tiled_overlay.png"
    )

    if path.exists():
        st.image(
            str(path),
            use_container_width=True
        )
    else:
        st.warning("Predicted overlay not found.")

with col2:
    st.subheader("Predicted Road Mask")

    path = output_file(
        "predicted_tiled_mask.png"
    )

    if path.exists():
        st.image(
            str(path),
            use_container_width=True
        )
    else:
        st.warning("Predicted mask not found.")


col3, col4 = st.columns(2)

with col3:
    st.subheader("Repaired Road Mask")

    path = output_file(
        "reconstruction_repaired.png"
    )

    if path.exists():
        st.image(
            str(path),
            use_container_width=True
        )
    else:
        st.warning("Repaired mask not found.")

with col4:
    st.subheader("Predicted Road Network Graph")

    path = output_file(
        "ground_truth_graph.png"
    )

    if path.exists():
        st.image(
            str(path),
            use_container_width=True
        )
    else:
        st.warning("Road graph not found.")


st.header("Model Evaluation")

metrics_path = output_file(
    "test_metrics.csv"
)

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)

    average_iou = metrics["iou"].mean()
    average_dice = metrics["dice"].mean()

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Average Test IoU",
            f"{average_iou:.2%}"
        )

    with metric_col2:
        st.metric(
            "Average Test Dice",
            f"{average_dice:.2%}"
        )

    st.subheader("Per-Image Test Results")

    st.dataframe(
        metrics,
        use_container_width=True
    )

    st.subheader("IoU and Dice by Test Image")

    chart_data = metrics.set_index(
        "image"
    )[["iou", "dice"]]

    st.bar_chart(chart_data)

else:
    st.warning(
        "Run evaluate_all.py before viewing test metrics."
    )


st.header("Resilience Analysis")

results_path = output_file(
    "resilience_results.csv"
)

if results_path.exists():
    results = pd.read_csv(results_path)

    results.columns = [
        column.strip()
        .lower()
        .replace(" ", "_")
        for column in results.columns
    ]

    st.subheader("Failure Simulation Results")

    st.dataframe(
        results,
        use_container_width=True
    )

    if {
        "scenario",
        "failure_rate",
        "resilience"
    }.issubset(results.columns):

        chart_data = results.pivot(
            index="failure_rate",
            columns="scenario",
            values="resilience"
        )

        st.subheader(
            "Random Failure vs Targeted Failure"
        )

        st.line_chart(chart_data)

else:
    st.warning(
        "Run simulate_failures.py before viewing resilience results."
    )


st.header("Current Prototype Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.metric(
        "Total Dataset Images",
        "40"
    )

with status_col2:
    st.metric(
        "Test Images",
        "6"
    )

with status_col3:
    st.metric(
        "Graph Components",
        "114"
    )


st.warning(
    "This is a research prototype. The predicted road graph is "
    "fragmented, so resilience values should not yet be interpreted "
    "as actual city-wide measurements."
)


st.header("Complete Development Pipeline")

st.markdown(
    """
1. Satellite image acquisition  
2. Image normalization and preprocessing  
3. U-Net road segmentation  
4. Probability thresholding  
5. Gap and occlusion repair  
6. Road graph construction  
7. Junction and critical-node analysis  
8. Random and targeted failure simulation  
9. Resilience measurement
"""
)