# ======================= Streamlit ML Dashboard =======================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ======================= Page Config =======================
st.set_page_config(page_title="ML App", layout="wide", page_icon="📊")

# ======================= Sidebar =======================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "EDA", "Outliers", "Univariate Analysis", "Bivariate Analysis", "Supervised Learning"])

# ======================= Main Page Title =======================
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 ML App</h1>", unsafe_allow_html=True)

# ======================= File Upload =======================
st.subheader("📂 Upload Dataset (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded!")
    st.dataframe(df.head())

# ======================= Main App Logic =======================
if df is not None:
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # ======================= DASHBOARD =======================
    if menu == "Dashboard":
        st.title("📊 ML Dashboard Overview")
        st.subheader("🔹 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Numeric Features", len(numeric_cols))
        col4.metric("Categorical Features", len(categorical_cols))

        # Missing Values
        st.subheader("🟠 Missing Values")
        missing = df.isnull().sum()
        fig_missing = px.bar(x=missing.index, y=missing.values, color=missing.values, 
                             color_continuous_scale='Oranges', text=missing.values)
        st.plotly_chart(fig_missing, use_container_width=True)

        # Numeric Feature Overview
        st.subheader("🔵 Numeric Feature Summary")
        if numeric_cols:
            fig = go.Figure()
            for col in numeric_cols:
                fig.add_trace(go.Box(y=df[col], name=col, boxmean=True, marker_color='skyblue'))
            fig.update_layout(title="Boxplots of Numeric Features", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Categorical Feature Overview
        st.subheader("🟢 Categorical Feature Distribution")
        for col in categorical_cols:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, 'count']
            fig = px.bar(counts, x=col, y='count', color=col, 
                         color_discrete_sequence=px.colors.qualitative.Pastel, text='count')
            st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.subheader("🔷 Correlation Heatmap")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

    # ======================= EDA =======================
    if menu == "EDA":
        st.title("🔹 Exploratory Data Analysis")
        st.subheader("Head of Dataset")
        st.dataframe(df.head())

        st.subheader("Shape")
        st.write(f"Rows: {df.shape[0]} \nColumns: {df.shape[1]}")

        st.subheader("Columns")
        st.write(df.columns.tolist())

        st.subheader("Describe (Numeric Features)")
        st.dataframe(df.describe())

        st.subheader("Basic Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values,
            "Percentage": (missing.values / len(df) * 100)
        })
        st.dataframe(missing_df)

        if categorical_cols:
            st.subheader("Categorical Feature Value Counts")
            for col in categorical_cols:
                st.markdown(f"**{col}**")
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, 'Count']
                st.dataframe(vc)

    # ======================= OUTLIERS =======================
    if menu == "Outliers":
        st.title("⚠️ Outlier Detection")
        outlier_dict = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            outlier_dict[col] = len(outliers)
        st.write("Number of outliers per numeric column:", outlier_dict)

        if st.button("Remove Outliers"):
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
            st.success("✅ Outliers removed!")
            st.dataframe(df.head())

    # ======================= UNIVARIATE ANALYSIS =======================
    if menu == "Univariate Analysis":
        st.title("🔹 Univariate Analysis")
        st.subheader("Numeric Features")
        for col in numeric_cols:
            fig = px.histogram(df, x=col, nbins=30, marginal="box", color_discrete_sequence=px.colors.sequential.Viridis)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 📝 Interpretation / Notes")
            st.text_area(f"Interpretation for {col}", "", height=120)

        st.subheader("Categorical Features")
        for col in categorical_cols:
            counts = df[col].value_counts().reset_index()
            counts.columns = [col, 'count']
            fig = px.bar(counts, x=col, y='count', color=col, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 📝 Interpretation / Notes")
            st.text_area(f"Interpretation for {col}", "", height=120)

    # ======================= BIVARIATE ANALYSIS (MULTI-PLOT) =======================
    if menu == "Bivariate Analysis":
        st.title("📊 Bivariate Analysis (Select Features)")

        all_cols = df.columns.tolist()
        x_cols = st.multiselect("Select Feature(s) for X-axis", all_cols, default=[all_cols[0]])
        y_cols = st.multiselect("Select Feature(s) for Y-axis / Target", all_cols, default=[all_cols[1]])

        for x_col in x_cols:
            for y_col in y_cols:
                if x_col == y_col:
                    continue

                st.subheader(f"📌 {x_col} vs {y_col}")

                # Numeric/Numeric -> Scatter
                if df[x_col].dtype in ["int64", "float64"] and df[y_col].dtype in ["int64", "float64"]:
                    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                                     title=f"Scatter Plot: {x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

                # Numeric/Categorical -> Boxplot
                elif df[x_col].dtype in ["int64", "float64"] and df[y_col].dtype == "object":
                    fig = px.box(df, x=y_col, y=x_col, points="all", color=y_col,
                                 title=f"Boxplot: {x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

                # Categorical/Numeric -> Boxplot
                elif df[x_col].dtype == "object" and df[y_col].dtype in ["int64", "float64"]:
                    fig = px.box(df, x=x_col, y=y_col, points="all", color=x_col,
                                 title=f"Boxplot: {x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

                # Categorical/Categorical -> Stacked Bar
                else:
                    ct = pd.crosstab(df[x_col], df[y_col]).reset_index()
                    ct_melt = ct.melt(id_vars=[x_col], value_vars=ct.columns[1:],
                                      var_name=y_col, value_name="Count")
                    fig = px.bar(ct_melt, x=x_col, y="Count", color=y_col, barmode='stack',
                                 title=f"Stacked Bar: {x_col} vs {y_col}")
                    st.plotly_chart(fig, use_container_width=True)

                # Interpretation Box
                st.markdown("### 📝 Interpretation / Notes")
                st.text_area(f"Add your interpretation for {x_col} vs {y_col}", "", height=120)

                # Summary Table
                st.subheader("📈 Summary")
                if df[y_col].dtype in ["int64", "float64"]:
                    st.dataframe(df.groupby(x_col)[y_col].describe())
                else:
                    st.dataframe(pd.crosstab(df[x_col], df[y_col], normalize="index"))

    # ======================= SUPERVISED LEARNING =======================
    if menu == "Supervised Learning":
        st.title("🔹 Model Training & Prediction")

        target = st.selectbox("Select Target Column", df.columns)
        features = st.multiselect("Select Feature Columns (leave empty for all except target)",
                                  [col for col in df.columns if col != target])
        if not features:
            features = [col for col in df.columns if col != target]

        X = df[features]
        y = df[target]

        numeric_X = X.select_dtypes(include=['int64','float64']).columns.tolist()
        categorical_X = X.select_dtypes(include=['object']).columns.tolist()

        if categorical_X:
            X = pd.get_dummies(X, columns=categorical_X)

        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        available_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC()
        }

        selected_models = st.multiselect("Select Models to Train", options=list(available_models.keys()),
                                         default=list(available_models.keys()))
        threshold = st.number_input("Overfitting Threshold (Train-Test Gap)", min_value=0.01, max_value=1.0,
                                    value=0.10, step=0.01)

        if st.button("Train Models"):
            results = {}
            best_model_name = None
            best_accuracy = 0

            for name in selected_models:
                model = available_models[name]
                X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()
                if name in ["Logistic Regression", "KNN", "SVM"]:
                    scaler = StandardScaler()
                    X_train_scaled[numeric_X] = scaler.fit_transform(X_train_scaled[numeric_X])
                    X_test_scaled[numeric_X] = scaler.transform(X_test_scaled[numeric_X])
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                train_acc = model.score(X_train_scaled, y_train)
                test_acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                results[name] = {
                    "Train Accuracy": train_acc,
                    "Test Accuracy": test_acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Confusion Matrix": confusion_matrix(y_test, y_pred),
                    "Model Object": model
                }

                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_model_name = name

            # Show metrics
            perf_df = pd.DataFrame({
                "Train Accuracy": [results[m]["Train Accuracy"] for m in results],
                "Test Accuracy": [results[m]["Test Accuracy"] for m in results],
                "Precision": [results[m]["Precision"] for m in results],
                "Recall": [results[m]["Recall"] for m in results],
                "F1 Score": [results[m]["F1 Score"] for m in results]
            }, index=results.keys())

            st.subheader("📈 Model Performance")
            fig = px.bar(perf_df, barmode='group', title="Metrics Comparison", color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✅ Best Model: {best_model_name} | Test Accuracy: {best_accuracy:.4f}")

            # Prediction Section
            st.subheader("🎯 Predict on New Data")
            best_model = results[best_model_name]["Model Object"]

            new_data = {}
            for col in features:
                if col in numeric_X:
                    new_data[col] = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
                else:
                    new_data[col] = st.text_input(f"Enter value for {col}", value=str(df[col].mode()[0]))

            if st.button("Predict"):
                input_df = pd.DataFrame([new_data])
                # Handle categorical encoding if needed
                if categorical_X:
                    input_df = pd.get_dummies(input_df)
                    for c in X_train.columns:
                        if c not in input_df.columns:
                            input_df[c] = 0
                    input_df = input_df[X_train.columns]

                if best_model_name in ["Logistic Regression", "KNN", "SVM"]:
                    input_df[numeric_X] = scaler.transform(input_df[numeric_X])

                prediction = best_model.predict(input_df)
                if y.dtype == 'object':
                    prediction = le_target.inverse_transform(prediction)
                st.success(f"Predicted {target}: {prediction[0]}")
