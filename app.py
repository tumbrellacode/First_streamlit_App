# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
)


# ── Data loading ────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/cleaned_airline_passenger_satisfaction.csv")


# ── Feature encoding ────────────────────────────────────────────────────────────

def prepare_modeling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["customer_type"] = df["customer_type"].map(
        {"Loyal Customer": 1, "disloyal Customer": 0}
    )
    df["type_of_travel"] = df["type_of_travel"].map(
        {"Business travel": 1, "Personal Travel": 0}
    )
    df["class_business"] = (df["class"] == "Business").astype(int)
    df["class_eco_plus"] = (df["class"] == "Eco Plus").astype(int)
    return df


# ── Cached model for Predictions page ──────────────────────────────────────────
# @st.cache_resource keeps the trained model in memory so it is only trained
# once, no matter how many times the user moves a slider.

@st.cache_resource
def train_prediction_model(_df: pd.DataFrame):
    df_model = prepare_modeling_dataframe(_df)
    X = df_model.drop(
        columns=[
            "satisfaction",
            "satisfaction_binary",
            "class",
            "arrival_delay_in_minutes",
            "is_short_haul",
        ]
    )
    y = df_model["satisfaction_binary"]
    model = RandomForestClassifier(
        max_depth=25,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    with st.spinner("Training prediction model for the first time… this takes about 30 seconds."):
        model.fit(X, y)
    return model, X.columns.tolist()


# ── Load data and navigation ────────────────────────────────────────────────────

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a Page",
    [
        "Home",
        "Data Overview",
        "Exploratory Data Analysis",
        "Model Training and Evaluation",
        "Make Predictions!",
        "Recommendations",
    ],
)


# ── HOME ────────────────────────────────────────────────────────────────────────

if page == "Home":
    st.title("✈️ Airline Passenger Satisfaction Project")

    st.write(
        """
        This project applies machine learning to predict whether an airline passenger
        is **satisfied** or **neutral/dissatisfied** based on their demographics,
        flight details, and ratings across 14 in-flight service categories.
        """
    )

    # Key metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Passengers", f"{df.shape[0]:,}")
    col2.metric("Features", df.shape[1])
    col3.metric("Best model accuracy", "96.2%")
    col4.metric("Top predictor", "Online boarding")

    st.subheader("Main Findings")
    st.write(
        """
        - **Random Forest** outperformed Logistic Regression on every metric
          (96.2% vs 87.3% test accuracy).
        - **Service-related features** were stronger predictors of satisfaction than delays.
        - **Digital touchpoints** (online boarding + Wi-Fi) account for over 32% of
          total feature importance.
        - Satisfaction patterns differed between **short-haul and non-short-haul** flights.
        """
    )

    st.subheader("How to use this app")
    st.write(
        """
        Use the navigation menu on the left to explore:

        - **Data Overview** — inspect the cleaned dataset.
        - **Exploratory Data Analysis** — key visual findings from Part 1.
        - **Model Training and Evaluation** — train and compare models interactively.
        - **Make Predictions!** — input passenger details and get a live prediction.
        - **Recommendations** — actionable insights for airlines.
        """
    )


# ── DATA OVERVIEW ───────────────────────────────────────────────────────────────

elif page == "Data Overview":
    st.title("📋 Data Overview")

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])

    st.subheader("Column names")
    st.write(df.columns.tolist())

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe())

    st.subheader("Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in the cleaned dataset.")
    else:
        st.dataframe(missing.rename("missing_count"))


# ── EXPLORATORY DATA ANALYSIS ───────────────────────────────────────────────────

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    import seaborn as sns

    st.write("Key visual findings from Part 1 of the project.")

    # Consistent font sizes across all EDA charts
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    # ── Satisfaction distribution ──
    st.subheader("Overall passenger satisfaction")

    counts = df["satisfaction"].value_counts().reindex(
        ["neutral or dissatisfied", "satisfied"]
    )
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.pie(
        counts,
        labels=["Neutral or dissatisfied", "Satisfied"],
        colors=["#FFDDAB", "#B7D7C0"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 13},
    )
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.write(
        """
        Most passengers were **neutral or dissatisfied (56.7%)**, while 43.3% were
        satisfied. This makes satisfaction a meaningful and slightly imbalanced target.
        """
    )

    # ── Flight distance ──
    st.subheader("Satisfaction by flight distance")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.histplot(
        data=df,
        x="flight_distance",
        hue="satisfaction",
        palette={"neutral or dissatisfied": "#FFAE37", "satisfied": "#7CB68D"},
        bins=30,
        alpha=0.6,
        ax=ax2,
    )
    ax2.set_xlabel("Flight Distance (km)")
    ax2.set_ylabel("Number of Passengers")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.write(
        """
        Passengers on **shorter flights are more often neutral or dissatisfied**.
        Longer flights show a higher concentration of satisfied passengers — which
        motivated the creation of the `is_short_haul` feature.
        """
    )

    # ── Service ratings ──
    st.subheader("Distribution of service feature ratings")

    service_cols = [
        "inflight_wifi_service", "departure_arrival_time_convenient",
        "ease_of_online_booking", "gate_location", "food_and_drink",
        "online_boarding", "seat_comfort", "inflight_entertainment",
        "on_board_service", "leg_room_service", "baggage_handling",
        "checkin_service", "inflight_service", "cleanliness",
    ]
    rating_order = [0, 1, 2, 3, 4, 5]
    distribution = pd.DataFrame(
        index=service_cols, columns=rating_order, dtype=float
    )
    for col in service_cols:
        distribution.loc[col] = (
            df[col].value_counts(normalize=True)
            .reindex(rating_order, fill_value=0)
            .mul(100)
            .values
        )
    distribution.index = [
        c.replace("_", " ").title() for c in distribution.index
    ]
    distribution = distribution.rename(columns={
        0: "0 = Not Applicable", 1: "1 = Very Poor", 2: "2 = Poor",
        3: "3 = Neutral", 4: "4 = Good", 5: "5 = Excellent",
    })

    fig3, ax3 = plt.subplots(figsize=(14, 8))
    distribution.plot(
        kind="barh",
        stacked=True,
        color=("#cfd2cd", "#c75146", "#ea8c55", "#F4C784", "#C5E17A", "#8FD19E"),
        ax=ax3,
        width=0.8,
    )
    ax3.set_xlabel("Percentage of Responses")
    ax3.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax3.invert_yaxis()
    ax3.legend(
        title="Rating",
        bbox_to_anchor=(0.5, 1.04),
        loc="lower center",
        ncol=6,
        frameon=False,
    )
    # Percentage labels inside each segment (only when >= 7%)
    for row_index, (_, row) in enumerate(distribution.iterrows()):
        cumulative = 0
        for value in row:
            if value >= 7:
                ax3.text(
                    cumulative + value / 2,
                    row_index,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
            cumulative += value
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.write(
        """
        Features such as **baggage handling, inflight service, and seat comfort**
        show a larger share of high ratings, while **inflight Wi-Fi and ease of
        online booking** show relatively more low ratings.
        """
    )

    # ── Correlation ──
    st.subheader("Correlation with satisfaction")

    corr_target = (
        df.corr(numeric_only=True)[["satisfaction_binary"]]
        .drop("satisfaction_binary")
        .sort_values("satisfaction_binary", ascending=False)
        .head(10)
    )
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.barh(
        corr_target.index[::-1],
        corr_target["satisfaction_binary"].values[::-1],
        color="#065A82",
    )
    ax4.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.write(
        """
        **Online boarding** has the strongest correlation (0.50). Notably,
        **delay variables** have near-zero correlation — passengers are more
        affected by service quality than by delays.
        """
    )


# ── MODEL TRAINING AND EVALUATION ───────────────────────────────────────────────

elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox(
        "Select a model",
        ["Logistic Regression", "Random Forest"],
    )

    df_model = prepare_modeling_dataframe(df)
    X = df_model.drop(
        columns=[
            "satisfaction",
            "satisfaction_binary",
            "class",
            "arrival_delay_in_minutes",
            "is_short_haul",
        ]
    )
    y = df_model["satisfaction_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, train_size=0.8, stratify=y
    )

    if st.button("Train Model"):
        if model_option == "Logistic Regression":
            scaler = StandardScaler()
            X_train_model = scaler.fit_transform(X_train)
            X_test_model = scaler.transform(X_test)
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            X_train_model = X_train
            X_test_model = X_test
            model = RandomForestClassifier(
                max_depth=25,
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )

        with st.spinner(f"Training {model_option}…"):
            model.fit(X_train_model, y_train)

        st.write(f"**Model selected:** {model_option}")

        col1, col2 = st.columns(2)
        col1.metric("Training accuracy", f"{model.score(X_train_model, y_train):.4f}")
        col2.metric("Test accuracy", f"{model.score(X_test_model, y_test):.4f}")

        st.subheader("Classification report")
        report = classification_report(
            y_test,
            model.predict(X_test_model),
            target_names=["Neutral / Dissatisfied", "Satisfied"],
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test_model,
            y_test,
            display_labels=["Neutral / Dissatisfied", "Satisfied"],
            ax=ax,
            cmap="Blues",
        )
        st.pyplot(fig)
        plt.close(fig)


# ── MAKE PREDICTIONS ────────────────────────────────────────────────────────────

elif page == "Make Predictions!":
    st.title("🎯 Make Predictions!")

    st.write(
        "Adjust the passenger profile and service ratings below. "
        "The model will instantly predict whether this passenger is likely to be satisfied."
    )

    # ── Passenger profile ──
    st.subheader("Passenger profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        customer_type = st.selectbox(
            "Customer type", ["disloyal Customer", "Loyal Customer"]
        )
    with col2:
        age = st.slider("Age", min_value=7, max_value=85, value=35)
        type_of_travel = st.selectbox(
            "Type of travel", ["Personal Travel", "Business travel"]
        )
    with col3:
        travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        flight_distance = st.slider(
            "Flight distance (km)", min_value=30, max_value=5000, value=1000
        )

    departure_delay_in_minutes = st.slider(
        "Departure delay (minutes)", min_value=0, max_value=1200, value=0
    )

    # ── Service ratings ──
    with st.expander("Service ratings (0 = Not applicable, 1–5 = Very poor to Excellent)"):
        col1, col2 = st.columns(2)
        with col1:
            inflight_wifi_service = st.slider("Inflight Wi-Fi service", 0, 5, 3)
            departure_arrival_time_convenient = st.slider("Departure/arrival time convenient", 0, 5, 3)
            ease_of_online_booking = st.slider("Ease of online booking", 0, 5, 3)
            gate_location = st.slider("Gate location", 0, 5, 3)
            food_and_drink = st.slider("Food and drink", 0, 5, 3)
            online_boarding = st.slider("Online boarding", 0, 5, 3)
            seat_comfort = st.slider("Seat comfort", 0, 5, 3)
        with col2:
            inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
            on_board_service = st.slider("On-board service", 0, 5, 3)
            leg_room_service = st.slider("Leg room service", 0, 5, 3)
            baggage_handling = st.slider("Baggage handling", 0, 5, 3)
            checkin_service = st.slider("Check-in service", 0, 5, 3)
            inflight_service = st.slider("Inflight service", 0, 5, 3)
            cleanliness = st.slider("Cleanliness", 0, 5, 3)

    # ── Build input dataframe ──
    user_input = pd.DataFrame({
        "gender": [gender],
        "customer_type": [customer_type],
        "age": [age],
        "type_of_travel": [type_of_travel],
        "class": [travel_class],
        "flight_distance": [flight_distance],
        "inflight_wifi_service": [inflight_wifi_service],
        "departure_arrival_time_convenient": [departure_arrival_time_convenient],
        "ease_of_online_booking": [ease_of_online_booking],
        "gate_location": [gate_location],
        "food_and_drink": [food_and_drink],
        "online_boarding": [online_boarding],
        "seat_comfort": [seat_comfort],
        "inflight_entertainment": [inflight_entertainment],
        "on_board_service": [on_board_service],
        "leg_room_service": [leg_room_service],
        "baggage_handling": [baggage_handling],
        "checkin_service": [checkin_service],
        "inflight_service": [inflight_service],
        "cleanliness": [cleanliness],
        "departure_delay_in_minutes": [departure_delay_in_minutes],
    })

    st.subheader("Your input values")
    st.dataframe(user_input)

    # ── Load cached model and predict ──
    model, feature_columns = train_prediction_model(df)

    user_input_model = prepare_modeling_dataframe(user_input)
    user_input_model = user_input_model.drop(columns=["class"])
    user_input_model = user_input_model.reindex(
        columns=feature_columns, fill_value=0
    )

    prediction = model.predict(user_input_model)[0]
    prediction_proba = model.predict_proba(user_input_model)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.success(
            f"✅ The model predicts this passenger is **Satisfied**  \n"
            f"Predicted probability of satisfaction: **{prediction_proba:.2%}**"
        )
        st.balloons()
    else:
        st.warning(
            f"⚠️ The model predicts this passenger is **Neutral or Dissatisfied**  \n"
            f"Predicted probability of satisfaction: **{prediction_proba:.2%}**"
        )


# ── RECOMMENDATIONS ─────────────────────────────────────────────────────────────

elif page == "Recommendations":
    st.title("💡 Recommendations")

    st.write(
        """
        Based on the feature importance results from the final Random Forest model,
        here are three actionable recommendations for airlines.
        """
    )

    # ── Feature importance chart ──
    st.subheader("Feature importance — top 10 predictors")

    feature_names = [
        "Online boarding", "Inflight Wi-Fi", "Type of travel",
        "Business class", "Inflight entertainment", "Seat comfort",
        "Online booking ease", "Customer type", "Flight distance",
        "Leg room service",
    ]
    importances = [0.175, 0.145, 0.100, 0.093, 0.060, 0.046, 0.038, 0.038, 0.038, 0.037]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#065A82" if i < 2 else "#1C7293" if i < 4 else "#5B8FA8"
              for i in range(len(feature_names))]
    ax.barh(feature_names[::-1], importances[::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 10 Feature Importances — Tuned Random Forest")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Dark blue = digital experience  |  Medium blue = travel context  |  Light blue = on-board comfort"
    )

    # ── Three recommendation cards ──
    st.subheader("From model to action")

    st.info(
        "**01 — Invest in the digital pre-flight experience**  \n"
        "*Feature importance: 0.175 (online boarding) + 0.038 (ease of online booking)*  \n\n"
        "Online boarding is the single most important predictor of satisfaction. "
        "Airlines should prioritise making mobile check-in and boarding passes seamless "
        "and reliable — especially on short-haul routes, where ease of online booking "
        "also ranks in the top 5."
    )

    st.info(
        "**02 — Treat inflight Wi-Fi as a core service, not an upsell**  \n"
        "*Feature importance: 0.145 overall — 0.206 for short-haul passengers*  \n\n"
        "Inflight Wi-Fi was the top predictor for short-haul passengers. "
        "Because in the real world, short-haul and non-short-haul flights often operate "
        "as different products with different service models and passenger expectations, "
        "airlines should consider bundling Wi-Fi into the ticket price on key routes "
        "or guaranteeing a minimum quality standard."
    )

    st.info(
        "**03 — Segment satisfaction strategy by travel type and cabin class**  \n"
        "*Feature importance: type of travel 0.100 + business class 0.093*  \n\n"
        "A one-size-fits-all approach will underperform. Short-haul passengers have a "
        "satisfaction rate of only 33% vs 58% for non-short-haul. Business travellers "
        "need connectivity and comfort; leisure passengers need value and entertainment. "
        "Airlines should run separate improvement programmes per segment."
    )
