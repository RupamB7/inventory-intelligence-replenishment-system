import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



# 1. PAGE SETUP

st.set_page_config(
    page_title="Inventory Intelligence System",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Inventory Intelligence & Demand Forecasting")
st.caption(
    "FreshRetailNet-50K | Moving Average + Linear Regression + Random Forest"
)

BASE_DIR = Path(__file__).parent
TRAIN_FILE = BASE_DIR / "train.parquet"
EVAL_FILE = BASE_DIR / "eval.parquet"

if not TRAIN_FILE.exists():
    st.error("train.parquet was not found in the same folder as app.py.")
    st.stop()

DATA_COLUMNS = [
    "city_id",
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level"
]


@st.cache_data(show_spinner=False)
def load_train_data(path):
    df = pd.read_parquet(path)

    keep = [c for c in DATA_COLUMNS if c in df.columns]
    df = df[keep].copy()

    df["dt"] = pd.to_datetime(df["dt"])
    return df


@st.cache_data(show_spinner=False)
def make_features(df):
    data = df[
        [
            "product_id",
            "store_id",
            "dt",
            "sale_amount"
        ]
    ].copy()

    data = data.sort_values(
        ["product_id", "store_id", "dt"]
    ).reset_index(drop=True)

    group = data.groupby(
        ["product_id", "store_id"],
        sort=False
    )

    data["previous_day_sales"] = group["sale_amount"].shift(1)

    data["rolling_7_sales"] = (
        group["sale_amount"]
        .transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
    )

    data["day_of_week"] = data["dt"].dt.dayofweek

    data = data.dropna(
        subset=["previous_day_sales", "rolling_7_sales"]
    )

    return data


train_raw = load_train_data(TRAIN_FILE)

with st.spinner("Preparing forecasting features..."):
    train_features = make_features(train_raw)

MODEL_FEATURES = [
    "previous_day_sales",
    "rolling_7_sales",
    "day_of_week",
    "product_id",
    "store_id"
]


@st.cache_resource(show_spinner=False)
def train_linear_model(data):
    model = LinearRegression()
    model.fit(
        data[MODEL_FEATURES],
        data["sale_amount"]
    )
    return model


@st.cache_resource(show_spinner=False)
def train_random_forest(data, sample_size):
    if len(data) > sample_size:
        sample = data.sample(
            n=sample_size,
            random_state=42
        )
    else:
        sample = data

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        sample[MODEL_FEATURES],
        sample["sale_amount"]
    )

    return model, len(sample)


def moving_average_forecast(history, horizon, window=7):
    values = list(history.astype(float))
    forecasts = []

    for _ in range(horizon):
        recent = values[-window:]

        if recent:
            prediction = float(np.mean(recent))
        else:
            prediction = 0.0

        prediction = max(0.0, prediction)
        forecasts.append(prediction)
        values.append(prediction)

    return np.array(forecasts)


def recursive_model_forecast(model, history, product_id, store_id, horizon):
    values = list(history["sale_amount"].astype(float))
    last_date = history["dt"].max()

    dates = []
    forecasts = []

    for day in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=day)

        recent = values[-7:]

        rolling_value = (
            float(np.mean(recent))
            if recent
            else 0.0
        )

        previous_value = (
            float(values[-1])
            if values
            else 0.0
        )

        row = pd.DataFrame([{
            "previous_day_sales": previous_value,
            "rolling_7_sales": rolling_value,
            "day_of_week": future_date.dayofweek,
            "product_id": product_id,
            "store_id": store_id
        }])

        prediction = float(
            model.predict(row[MODEL_FEATURES])[0]
        )

        prediction = max(0.0, prediction)

        dates.append(future_date)
        forecasts.append(prediction)
        values.append(prediction)

    return pd.DataFrame({
        "Date": dates,
        "Forecast": forecasts
    })


def metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "Bias": np.mean(predicted - actual)
    }
#Train Models
st.sidebar.header("⚙️ Settings")

rf_sample_size = st.sidebar.number_input(
    "Random Forest training rows",
    min_value=50_000,
    max_value=len(train_features),
    value=min(250_000, len(train_features)),
    step=50_000
)

with st.spinner("Training Linear Regression..."):
    linear_model = train_linear_model(train_features)

with st.spinner("Training Random Forest..."):
    random_forest, rf_rows = train_random_forest(
        train_features,
        int(rf_sample_size)
    )

# 6. DATASET OVERVIEW

st.header("📊 Dataset Overview")

number_of_days = (
    train_raw["dt"].max() -
    train_raw["dt"].min()
).days + 1

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Rows", f"{len(train_raw):,}")
c2.metric("Products", f"{train_raw['product_id'].nunique():,}")
c3.metric("Stores", f"{train_raw['store_id'].nunique():,}")
c4.metric("Cities", f"{train_raw['city_id'].nunique():,}")
c5.metric("Historical Days", number_of_days)

st.caption(
    f"Historical period: {train_raw['dt'].min().date()} "
    f"to {train_raw['dt'].max().date()}"
)

# 7. PRODUCT PERFORMANCE OVERVIEW

st.divider()
st.header("📦 Product Performance Overview")

st.write(
    "Use this section to see which products sell more, "
    "which have more observed stockout exposure, and "
    "which forecasting method has been more accurate."
)

run_product_overview = st.button(
    "🔎 Analyse All Products",
    type="primary"
)

if run_product_overview:

    with st.spinner(
        "Evaluating the three forecasting methods across the product portfolio..."
    ):

        # Sales performance
        product_sales = (
            train_raw
            .groupby("product_id")
            .agg(
                Total_Sales=("sale_amount", "sum"),
                Average_Daily_Sales=("sale_amount", "mean"),
                Observations=("sale_amount", "size")
            )
            .reset_index()
        )

        # Stockout exposure
        if "stock_hour6_22_cnt" in train_raw.columns:

            stock_data = train_raw[
                ["product_id", "stock_hour6_22_cnt"]
            ].copy()

            stock_data["stockout_day"] = (
                stock_data["stock_hour6_22_cnt"] > 0
            )

            product_stock = (
                stock_data
                .groupby("product_id")
                .agg(
                    Stockout_Days=("stockout_day", "sum"),
                    Observations=("stockout_day", "size")
                )
                .reset_index()
            )

            product_stock["Stockout_Day_Rate_%"] = (
                product_stock["Stockout_Days"]
                / product_stock["Observations"]
                * 100
            )

        else:

            product_stock = pd.DataFrame({
                "product_id": train_raw["product_id"].unique(),
                "Zero_Stock_Observations": np.nan,
                "Stock_Observations": np.nan,
                "Stockout_Rate_%": np.nan
            })

        # Last 14 historical observations for validation
        validation_days = 14

        group_size = train_features.groupby(
            ["product_id", "store_id"]
        )["dt"].transform("size")

        position_from_end = (
            group_size
            - train_features.groupby(
                ["product_id", "store_id"]
            ).cumcount()
        )

        validation_mask = position_from_end <= validation_days

        validation = train_features.loc[
            validation_mask
        ].copy()

        # Moving Average
        validation["MA_Prediction"] = (
            validation["rolling_7_sales"]
            .fillna(0)
            .clip(lower=0)
        )

        # Linear Regression
        validation["LR_Prediction"] = (
            linear_model
            .predict(validation[MODEL_FEATURES])
        )

        validation["LR_Prediction"] = (
            validation["LR_Prediction"]
            .clip(lower=0)
        )

        # Random Forest
        validation["RF_Prediction"] = (
            random_forest
            .predict(validation[MODEL_FEATURES])
        )

        validation["RF_Prediction"] = (
            validation["RF_Prediction"]
            .clip(lower=0)
        )

        # Product-level metrics
        def product_model_metrics(group):

            actual = group["sale_amount"].values
            result = {}

            for model_name, prediction_column in [
                ("Moving Average", "MA_Prediction"),
                ("Linear Regression", "LR_Prediction"),
                ("Random Forest", "RF_Prediction")
            ]:

                prediction = group[prediction_column].values

                result[f"{model_name}_MAE"] = (
                    mean_absolute_error(actual, prediction)
                )

                result[f"{model_name}_RMSE"] = (
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            prediction
                        )
                    )
                )

                result[f"{model_name}_Bias"] = (
                    np.mean(prediction - actual)
                )

            return pd.Series(result)

        product_metrics = (
            validation
            .groupby("product_id")
            .apply(product_model_metrics)
            .reset_index()
        )

        # Identify best model using lowest MAE
        mae_columns = {
            "Moving Average": "Moving Average_MAE",
            "Linear Regression": "Linear Regression_MAE",
            "Random Forest": "Random Forest_MAE"
        }

        product_metrics["Best_Model"] = (
            product_metrics[
                list(mae_columns.values())
            ]
            .idxmin(axis=1)
            .map({
                value: key
                for key, value in mae_columns.items()
            })
        )

        product_metrics["Best_MAE"] = (
            product_metrics[
                list(mae_columns.values())
            ].min(axis=1)
        )

        # Combine sales, stockout and forecasting results
        product_overview = (
            product_sales
            .merge(
                product_stock,
                on="product_id",
                how="left"
            )
            .merge(
                product_metrics,
                on="product_id",
                how="left"
            )
        )

        # Filters
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            minimum_sales = st.number_input(
                "Minimum total sales",
                min_value=0.0,
                value=0.0,
                step=100.0
            )

        with filter_col2:
            model_filter = st.selectbox(
                "Best forecasting model",
                [
                    "All",
                    "Moving Average",
                    "Linear Regression",
                    "Random Forest"
                ]
            )

        filtered_products = product_overview[
            product_overview["Total_Sales"]
            >= minimum_sales
        ].copy()

        if model_filter != "All":
            filtered_products = filtered_products[
                filtered_products["Best_Model"]
                == model_filter
            ]

        # Portfolio KPIs
        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Products Shown",
            f"{len(filtered_products):,}"
        )

        highest_sales_product = product_overview.loc[
            product_overview["Total_Sales"].idxmax(),
            "product_id"
        ]

        p2.metric(
            "Highest-Selling Product",
            str(highest_sales_product)
        )

        if product_overview["Stockout_Rate_%"].notna().any():

            highest_stockout_product = product_overview.loc[
                product_overview["Stockout_Rate_%"].idxmax(),
                "product_id"
            ]

            p3.metric(
                "Highest Stockout Exposure",
                str(highest_stockout_product)
            )

        else:

            p3.metric(
                "Stockout Data",
                "Unavailable"
            )

        p4.metric(
            "Lowest Product MAE",
            f"{product_overview['Best_MAE'].min():.4f}"
        )

        # Top products by sales
        st.subheader("Top Products by Historical Sales")

        top_sales = (
            product_overview
            .sort_values(
                "Total_Sales",
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            top_sales[
                [
                    "product_id",
                    "Total_Sales",
                    "Average_Daily_Sales",
                    "Stockout_Rate_%",
                    "Best_Model",
                    "Best_MAE"
                ]
            ].style.format({
                "Total_Sales": "{:,.2f}",
                "Average_Daily_Sales": "{:,.2f}",
                "Stockout_Rate_%": "{:.2f}%",
                "Best_MAE": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Highest stockout exposure
        st.subheader(
            "Products with Highest Observed Stockout Exposure"
        )

        if product_overview["Stockout_Rate_%"].notna().any():

            top_stockout = (
                product_overview
                .sort_values(
                    "Stockout_Rate_%",
                    ascending=False
                )
                .head(20)
            )

            st.dataframe(
                top_stockout[
                    [
                        "product_id",
                        "Total_Sales",
                        "Stockout_Rate_%",
                        "Zero_Stock_Observations",
                        "Best_Model",
                        "Best_MAE"
                    ]
                ].style.format({
                    "Total_Sales": "{:,.2f}",
                    "Stockout_Rate_%": "{:.2f}%",
                    "Best_MAE": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

        # Highest forecasting error
        st.subheader(
            "Products with Highest Forecasting Error"
        )

        worst_forecast = (
            product_overview
            .sort_values(
                "Best_MAE",
                ascending=False
            )
            .head(20)
        )

        st.dataframe(
            worst_forecast[
                [
                    "product_id",
                    "Total_Sales",
                    "Best_Model",
                    "Best_MAE",
                    "Moving Average_Bias",
                    "Linear Regression_Bias",
                    "Random Forest_Bias"
                ]
            ].style.format({
                "Total_Sales": "{:,.2f}",
                "Best_MAE": "{:.4f}",
                "Moving Average_Bias": "{:.4f}",
                "Linear Regression_Bias": "{:.4f}",
                "Random Forest_Bias": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Complete product table
        st.subheader(
            "Complete Product Performance Table"
        )

        st.dataframe(
            filtered_products[
                [
                    "product_id",
                    "Total_Sales",
                    "Average_Daily_Sales",
                    "Stockout_Rate_%",
                    "Moving Average_MAE",
                    "Linear Regression_MAE",
                    "Random Forest_MAE",
                    "Moving Average_Bias",
                    "Linear Regression_Bias",
                    "Random Forest_Bias",
                    "Best_Model"
                ]
            ].sort_values(
                "Total_Sales",
                ascending=False
            ).style.format({
                "Total_Sales": "{:,.2f}",
                "Average_Daily_Sales": "{:,.2f}",
                "Stockout_Rate_%": "{:.2f}%",
                "Moving Average_MAE": "{:.4f}",
                "Linear Regression_MAE": "{:.4f}",
                "Random Forest_MAE": "{:.4f}",
                "Moving Average_Bias": "{:.4f}",
                "Linear Regression_Bias": "{:.4f}",
                "Random Forest_Bias": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Forecasting metrics are calculated on the final "
            "14 historical observations for each product-store "
            "series. Product-level metrics combine the available "
            "store-level observations for that product. The best "
            "model is selected using the lowest MAE."
        )
        
# 8. PRODUCT + STORE FORECAST

st.divider()
st.header("🔎 Product & Store Forecast")

st.write(
    "Choose one, multiple, or all available product-store combinations "
    "and generate future demand forecasts."
)
# Forecast selection
selection_mode = st.radio(
    "Forecast selection",
    ["Single", "Multiple", "All"],
    horizontal=True,
    help=(
        "Single is useful for detailed drill-down. Multiple lets you "
        "forecast a selected group. All forecasts every available "
        "product-store combination."
    )
)

available_pairs = (
    train_raw[["product_id", "store_id"]]
    .drop_duplicates()
    .sort_values(["product_id", "store_id"])
)

available_products = sorted(
    available_pairs["product_id"].unique()
)

if selection_mode == "Single":

    selected_product = st.selectbox(
        "Product",
        available_products
    )

    available_stores = sorted(
        available_pairs.loc[
            available_pairs["product_id"] == selected_product,
            "store_id"
        ].unique()
    )

    selected_store = st.selectbox(
        "Store",
        available_stores
    )

    selected_pairs = pd.DataFrame({
        "product_id": [selected_product],
        "store_id": [selected_store]
    })

elif selection_mode == "Multiple":

    selected_products = st.multiselect(
        "Products",
        available_products,
        default=available_products[:1]
    )

    selected_stores = st.multiselect(
        "Stores",
        sorted(available_pairs["store_id"].unique()),
        default=[]
    )

    if selected_products and selected_stores:
        selected_pairs = available_pairs[
            available_pairs["product_id"].isin(selected_products)
            & available_pairs["store_id"].isin(selected_stores)
        ].copy()
    elif selected_products:
        selected_pairs = available_pairs[
            available_pairs["product_id"].isin(selected_products)
        ].copy()
    else:
        selected_pairs = available_pairs.iloc[0:0].copy()

else:

    selected_pairs = available_pairs.copy()

    st.info(
        f"All available combinations selected: "
        f"{len(selected_pairs):,} product-store combinations."
    )

forecast_horizon = st.slider(
    "Forecast horizon (days)",
    min_value=1,
    max_value=90,
    value=7
)

if selection_mode == "Single":
    st.caption("1 product-store combination selected.")
else:
    st.caption(
        f"{len(selected_pairs):,} product-store combinations selected."
    )

analyse_forecast = st.button(
    "🚀 Generate Forecast",
    type="primary"
)

if analyse_forecast:

    if selected_pairs.empty:
        st.warning("Please select at least one product or store combination.")
        st.stop()

   
     #Batch forecasting
    

    forecast_rows = []
    model_rows = []
    skipped_pairs = []

    progress = st.progress(0)
    status = st.empty()

    total_pairs = len(selected_pairs)

    with st.spinner(
        f"Generating forecasts for {total_pairs:,} product-store combination(s)..."
    ):

        for pair_number, pair in enumerate(
            selected_pairs.itertuples(index=False),
            start=1
        ):

            product_id = pair.product_id
            store_id = pair.store_id

            series = train_raw[
                (
                    train_raw["product_id"] == product_id
                )
                &
                (
                    train_raw["store_id"] == store_id
                )
            ].copy()

            series = series.sort_values("dt")

            if len(series) < 20:
                skipped_pairs.append({
                    "product_id": product_id,
                    "store_id": store_id,
                    "reason": "Less than 20 historical observations"
                })
                continue

            series_features = train_features[
                (
                    train_features["product_id"] == product_id
                )
                &
                (
                    train_features["store_id"] == store_id
                )
            ].copy()

            if len(series_features) < 7:
                skipped_pairs.append({
                    "product_id": product_id,
                    "store_id": store_id,
                    "reason": "Not enough engineered observations"
                })
                continue

            validation_days = min(
                14,
                max(7, len(series_features) // 5)
            )

            validation = series_features.tail(validation_days).copy()
            actual = validation["sale_amount"].values

            ma_prediction = (
                validation["rolling_7_sales"]
                .fillna(0)
                .clip(lower=0)
                .values
            )

            lr_prediction = np.maximum(
                0,
                linear_model.predict(validation[MODEL_FEATURES])
            )

            rf_prediction = np.maximum(
                0,
                random_forest.predict(validation[MODEL_FEATURES])
            )

            comparison = pd.DataFrame([
                {
                    "Model": "Moving Average",
                    **metrics(actual, ma_prediction)
                },
                {
                    "Model": "Linear Regression",
                    **metrics(actual, lr_prediction)
                },
                {
                    "Model": "Random Forest",
                    **metrics(actual, rf_prediction)
                }
            ])

            best_row = comparison.loc[
                comparison["MAE"].idxmin()
            ]

            best_model = best_row["Model"]

            ma_future = moving_average_forecast(
                series["sale_amount"],
                forecast_horizon
            )

            lr_future = recursive_model_forecast(
                linear_model,
                series,
                product_id,
                store_id,
                forecast_horizon
            )

            rf_future = recursive_model_forecast(
                random_forest,
                series,
                product_id,
                store_id,
                forecast_horizon
            )

            for i in range(forecast_horizon):
                forecast_rows.append({
                    "Product": product_id,
                    "Store": store_id,
                    "Date": lr_future.loc[i, "Date"],
                    "Moving Average": ma_future[i],
                    "Linear Regression": lr_future.loc[i, "Forecast"],
                    "Random Forest": rf_future.loc[i, "Forecast"],
                    "Selected Model": {
                        "Moving Average": ma_future[i],
                        "Linear Regression": lr_future.loc[i, "Forecast"],
                        "Random Forest": rf_future.loc[i, "Forecast"]
                    }[best_model]
                })

            model_rows.append({
                "Product": product_id,
                "Store": store_id,
                "Total Historical Sales": series["sale_amount"].sum(),
                "Average Daily Sales": series["sale_amount"].mean(),
                "Best Model": best_model,
                "Best MAE": best_row["MAE"],
                "Best RMSE": best_row["RMSE"],
                "Best Bias": best_row["Bias"]
            })

            progress.progress(
                pair_number / total_pairs
            )
            status.text(
                f"Processed {pair_number:,} of {total_pairs:,} combinations..."
            )

    progress.empty()
    status.empty()

    if not model_rows:
        st.error("No selected product-store combinations had enough data to forecast.")
        st.stop()

    model_summary = pd.DataFrame(model_rows)
    batch_forecast = pd.DataFrame(forecast_rows)

    # Summary
    
    st.subheader("📊 Forecast Summary")

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Combinations Forecasted",
        f"{len(model_summary):,}"
    )

    s2.metric(
        "Forecast Days",
        f"{forecast_horizon}"
    )

    s3.metric(
        "Combinations Skipped",
        f"{len(skipped_pairs):,}"
    )

    st.dataframe(
        model_summary.style.format({
            "Total Historical Sales": "{:,.2f}",
            "Average Daily Sales": "{:,.2f}",
            "Best MAE": "{:.4f}",
            "Best RMSE": "{:.4f}",
            "Best Bias": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Forecast results
    st.subheader("🔮 Future Forecasts")

    st.dataframe(
        batch_forecast.style.format({
            "Moving Average": "{:.2f}",
            "Linear Regression": "{:.2f}",
            "Random Forest": "{:.2f}",
            "Selected Model": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Download batch forecast
    csv_data = batch_forecast.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Forecast CSV",
        data=csv_data,
        file_name="batch_forecasts.csv",
        mime="text/csv"
    )

   
    # Detailed view for single selection


    if selection_mode == "Single" and len(model_summary) == 1:

        product_id = model_summary.iloc[0]["Product"]
        store_id = model_summary.iloc[0]["Store"]
        best_model = model_summary.iloc[0]["Best Model"]

        series = train_raw[
            (
                train_raw["product_id"] == product_id
            )
            &
            (
                train_raw["store_id"] == store_id
            )
        ].copy().sort_values("dt")

        st.subheader(
            f"Product {product_id} — Store {store_id}"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Total Historical Sales",
            f"{series['sale_amount'].sum():,.2f}"
        )

        b.metric(
            "Average Daily Sales",
            f"{series['sale_amount'].mean():,.2f}"
        )

        c.metric(
            "Historical Observations",
            f"{len(series):,}"
        )

        st.line_chart(
            series.set_index("dt")["sale_amount"]
        )

        # Detailed model comparison
        series_features = train_features[
            (
                train_features["product_id"] == product_id
            )
            &
            (
                train_features["store_id"] == store_id
            )
        ].copy()

        validation_days = min(
            14,
            max(7, len(series_features) // 5)
        )

        validation = series_features.tail(validation_days).copy()
        actual = validation["sale_amount"].values

        ma_prediction = (
            validation["rolling_7_sales"]
            .fillna(0)
            .clip(lower=0)
            .values
        )

        lr_prediction = np.maximum(
            0,
            linear_model.predict(validation[MODEL_FEATURES])
        )

        rf_prediction = np.maximum(
            0,
            random_forest.predict(validation[MODEL_FEATURES])
        )

        comparison = pd.DataFrame([
            {
                "Model": "Moving Average",
                **metrics(actual, ma_prediction)
            },
            {
                "Model": "Linear Regression",
                **metrics(actual, lr_prediction)
            },
            {
                "Model": "Random Forest",
                **metrics(actual, rf_prediction)
            }
        ])

        st.subheader("🏆 Model Comparison")

        st.dataframe(
            comparison.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "Bias": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"Current best-performing model by MAE: **{best_model}**"
        )

        detailed_forecast = batch_forecast.copy()

        st.subheader(
            f"🔮 Next {forecast_horizon} Days"
        )

        st.line_chart(
            detailed_forecast.set_index("Date")[
                [
                    "Moving Average",
                    "Linear Regression",
                    "Random Forest",
                    "Selected Model"
                ]
            ]
        )

        st.subheader("Forecast Totals")

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Moving Average",
            f"{detailed_forecast['Moving Average'].sum():,.2f}"
        )

        q2.metric(
            "Linear Regression",
            f"{detailed_forecast['Linear Regression'].sum():,.2f}"
        )

        q3.metric(
            "Random Forest",
            f"{detailed_forecast['Random Forest'].sum():,.2f}"
        )

        q4.metric(
            "Selected Model",
            f"{detailed_forecast['Selected Model'].sum():,.2f}"
        )

        # Stockout analysis
        st.subheader("⚠️ Stockout Analysis")

        if "stock_hour6_22_cnt" in series.columns:

            stockout_days = (
                series["stock_hour6_22_cnt"] > 0
            ).sum()

            total_out_of_stock_hours = (
                series["stock_hour6_22_cnt"]
                .fillna(0)
                .clip(lower=0)
                .sum()
            )

            stockout_day_rate = (
                stockout_days / len(series) * 100
            )

            s1, s2, s3 = st.columns(3)

            s1.metric(
                "Days With Stockout Hours",
                f"{stockout_days:,}"
            )

            s2.metric(
                "Stockout-Day Rate",
                f"{stockout_day_rate:.2f}%"
            )

            s3.metric(
                "Total Out-of-Stock Hours",
                f"{total_out_of_stock_hours:,.0f}"
            )

            st.caption(
                "stock_hour6_22_cnt is the number of out-of-stock hours "
                "between 06:00 and 22:00. It is not current on-hand inventory."
            )

        # Discount/activity
        st.subheader("🏷️ Discount & Activity")

        if "discount" in series.columns:

            discounted_sales = series.loc[
                series["discount"] > 0,
                "sale_amount"
            ].mean()

            normal_sales = series.loc[
                series["discount"] == 0,
                "sale_amount"
            ].mean()

            d1, d2 = st.columns(2)

            d1.metric(
                "Average Sales With Discount",
                f"{discounted_sales:.2f}"
            )

            d2.metric(
                "Average Sales Without Discount",
                f"{normal_sales:.2f}"
            )

        if "activity_flag" in series.columns:

            activity_sales = series.loc[
                series["activity_flag"] == 1,
                "sale_amount"
            ].mean()

            no_activity_sales = series.loc[
                series["activity_flag"] == 0,
                "sale_amount"
            ].mean()

            d3, d4 = st.columns(2)

            d3.metric(
                "Average Sales During Activity",
                f"{activity_sales:.2f}"
            )

            d4.metric(
                "Average Sales Without Activity",
                f"{no_activity_sales:.2f}"
            )

        # Weather
        st.subheader("🌦️ Weather")

        weather_columns = [
            c for c in [
                "precpt",
                "avg_temperature",
                "avg_humidity",
                "avg_wind_level"
            ]
            if c in series.columns
        ]

        if weather_columns:

            weather = (
                series[weather_columns]
                .mean()
                .to_frame("Average")
            )

            st.dataframe(
                weather.style.format("{:.2f}"),
                use_container_width=True
            )

    if skipped_pairs:
        with st.expander(
            f"⚠️ Skipped combinations ({len(skipped_pairs):,})"
        ):
            st.dataframe(
                pd.DataFrame(skipped_pairs),
                use_container_width=True,
                hide_index=True
            )


# 9. UNSEEN EVALUATION


st.divider()
st.header("🧪 Unseen Evaluation")

st.write(
    "Evaluate all three forecasting methods on a separate "
    "evaluation dataset that was not used for model training."
)

if EVAL_FILE.exists():

    st.info(
        "eval.parquet was found in the project folder."
    )

    run_unseen = st.button(
        "▶️ Run Unseen Evaluation",
        type="primary"
    )

    if run_unseen:

        with st.spinner("Loading evaluation data..."):

            eval_raw = pd.read_parquet(EVAL_FILE)
            eval_raw["dt"] = pd.to_datetime(eval_raw["dt"])

        required = [
            "product_id",
            "store_id",
            "dt",
            "sale_amount"
        ]

        missing = [
            c for c in required
            if c not in eval_raw.columns
        ]

        if missing:

            st.error(
                "Evaluation data is missing: "
                + ", ".join(missing)
            )

            st.stop()

        combined = pd.concat(
            [
                train_raw[
                    [
                        "product_id",
                        "store_id",
                        "dt",
                        "sale_amount"
                    ]
                ],
                eval_raw[
                    [
                        "product_id",
                        "store_id",
                        "dt",
                        "sale_amount"
                    ]
                ]
            ],
            ignore_index=True
        )

        combined = combined.sort_values(
            [
                "product_id",
                "store_id",
                "dt"
            ]
        )

        group = combined.groupby(
            ["product_id", "store_id"],
            sort=False
        )

        combined["previous_day_sales"] = (
            group["sale_amount"].shift(1)
        )

        combined["rolling_7_sales"] = (
            group["sale_amount"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(7, min_periods=1)
                .mean()
            )
        )

        combined["day_of_week"] = (
            combined["dt"].dt.dayofweek
        )

        eval_features = combined.merge(
            eval_raw[
                [
                    "product_id",
                    "store_id",
                    "dt"
                ]
            ],
            on=[
                "product_id",
                "store_id",
                "dt"
            ],
            how="inner"
        )

        eval_features = eval_features.drop_duplicates(
            [
                "product_id",
                "store_id",
                "dt"
            ]
        )

        eval_features = eval_features.dropna(
            subset=[
                "previous_day_sales",
                "rolling_7_sales"
            ]
        )

        actual = eval_features[
            "sale_amount"
        ].values

        ma_prediction = (
            eval_features["rolling_7_sales"]
            .clip(lower=0)
            .values
        )

        lr_prediction = (
            linear_model
            .predict(eval_features[MODEL_FEATURES])
        )
        lr_prediction = np.maximum(0, lr_prediction)

        rf_prediction = (
            random_forest
            .predict(eval_features[MODEL_FEATURES])
        )
        rf_prediction = np.maximum(0, rf_prediction)

        unseen_results = pd.DataFrame([
            {
                "Model": "Moving Average",
                **metrics(actual, ma_prediction)
            },
            {
                "Model": "Linear Regression",
                **metrics(actual, lr_prediction)
            },
            {
                "Model": "Random Forest",
                **metrics(actual, rf_prediction)
            }
        ])

        st.subheader("📊 Unseen Evaluation Results")

        st.dataframe(
            unseen_results.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "Bias": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        total_actual = actual.sum()

        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "Actual Sales",
            f"{total_actual:,.2f}"
        )

        e2.metric(
            "Moving Average",
            f"{ma_prediction.sum():,.2f}"
        )

        e3.metric(
            "Linear Regression",
            f"{lr_prediction.sum():,.2f}"
        )

        e4.metric(
            "Random Forest",
            f"{rf_prediction.sum():,.2f}"
        )

        best_unseen = unseen_results.loc[
            unseen_results["MAE"].idxmin()
        ]

        st.success(
            "Best-performing model on this unseen evaluation "
            f"set by MAE: **{best_unseen['Model']}**"
        )

        st.caption(
            f"Evaluation rows used: {len(eval_features):,}"
        )

else:

    st.warning(
        "eval.parquet was not found. Put it in the same folder "
        "as app.py."
    )

# 10. DATASET LIMITATIONS

st.divider()

st.subheader("ℹ️ Dataset Limitations")

st.write(
    "FreshRetailNet-50K does not provide real supplier records, "
    "purchase orders, lead times, MOQ, expiry dates or a real-time "
    "current on-hand inventory ledger. These are therefore not "
    "fabricated in this version."
)
