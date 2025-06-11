import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Updated import
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u

# Streamlit App
st.title("Orbital Optimization for Efficient Satellite Mission Planning Using Machine Learning")

# File Upload Section
st.sidebar.header("Upload Datasets")
uploaded_input = st.sidebar.file_uploader("Upload Input Dataset (CSV)", type="csv")
uploaded_features = st.sidebar.file_uploader("Upload Features Dataset (CSV)", type="csv")

if uploaded_input and uploaded_features:
    # Load datasets
    df_input = pd.read_csv(uploaded_input)
    df_features = pd.read_csv(uploaded_features)

    # Merge the datasets
    df = pd.concat([df_input, df_features], axis=1)

    # Features (X) and Target (y)
    X = df[['MEAN_MOTION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'MEAN_MOTION_DOT']].values
    y = np.array([
        [6371 + np.random.rand() * 1000, np.random.rand() * 0.1, np.random.rand() * 180]
        for _ in range(df.shape[0])
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Performance
    mse = mean_squared_error(y_test, y_pred)
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"Mean Squared Error: {mse:.4f}")

    # Predict New Input
    st.sidebar.subheader("Predict New Orbit Parameters")
    mean_motion = st.sidebar.number_input("Mean Motion", value=0.9)
    ra_of_asc_node = st.sidebar.number_input("RA of Ascending Node", value=0.4)
    arg_of_pericenter = st.sidebar.number_input("Argument of Pericenter", value=0.3)
    mean_anomaly = st.sidebar.number_input("Mean Anomaly", value=0.2)
    mean_motion_dot = st.sidebar.number_input("Mean Motion Dot", value=0.05)

    new_input = np.array([[mean_motion, ra_of_asc_node, arg_of_pericenter, mean_anomaly, mean_motion_dot]])
    predicted_orbit = model.predict(new_input)
    predicted_sma, predicted_ecc, predicted_inc = predicted_orbit[0]

    # Display Predictions
    st.subheader("Predicted Orbital Parameters")
    st.write(f"Semi-major axis: {predicted_sma:.2f} km")
    st.write(f"Eccentricity: {predicted_ecc:.6f}")
    st.write(f"Inclination: {predicted_inc:.2f} degrees")

    # Visualization
    try:
        semi_major_axis = predicted_sma * u.km
        eccentricity = predicted_ecc * u.one
        inclination = predicted_inc * u.deg

        orbit = Orbit.from_classical(
            Earth,
            semi_major_axis,
            eccentricity,
            inclination,
            0 * u.deg,  # RAAN
            0 * u.deg,  # Argument of Periapsis
            0 * u.deg,  # True Anomaly
        )

        time_of_flight = np.linspace(0, 1, 500) * orbit.period
        positions = np.array([orbit.propagate(t).r.to(u.km).value for t in time_of_flight])

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            name="Predicted Orbit",
            line=dict(color='green', width=2)
        ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=5, color='red'),
            name="Earth"
        ))

        fig.update_layout(
            title="Predicted Satellite Trajectory (3D)",
            scene=dict(
                xaxis_title="X Position (km)",
                yaxis_title="Y Position (km)",
                zaxis_title="Z Position (km)"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=True
                    )

        st.subheader("3D Orbital Visualization")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Visualization failed: {e}")

else:
    st.warning("Please upload both input and feature datasets to proceed.")