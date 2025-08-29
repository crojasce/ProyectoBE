    # app.py
#
# Streamlit app para proyecto: Predicción de reingreso (<30 días) en pacientes con diabetes
# - Cargar datos / Descripción del dataset
# - EDA
# - Preprocesamiento
# - Modelado (XGBoost y RandomForest)
# - Optimización de umbral
# - Explicabilidad SHAP (matplotlib-safe y force_plot HTML opcional)
# - Export de modelos
#
# Uso: streamlit run app.py
#

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score
)

sns.set_palette("pastel")
plt.style.use("ggplot")

st.set_page_config(layout="wide", page_title="Diabetes Reingreso - ML App")

# ------------------------
# Helpers: sanitizar columnas
# ------------------------
def sanitize_colname(name):
    s = re.sub(r"[\[\]\<\>\(\),\s%:/\\]", "_", str(name))
    s = re.sub(r"[^0-9a-zA-Z_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s == "":
        s = "col"
    return s

# ------------------------
# Cargar dataset (uploader)
# ------------------------
@st.cache_data
def load_csv_from_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def load_csv_from_path(path):
    df = pd.read_csv(path)
    return df

# ------------------------
# Preprocesamiento: función
# ------------------------
def preprocess(df, drop_cols=None, use_get_dummies=True):
    df = df.copy()
    # columnas a eliminar si existen
    if drop_cols is None:
        drop_cols = ["weight", "payer_code", "medical_specialty"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Reparar codificaciones
    if "race" in df.columns:
        df["race"] = df["race"].replace("?", "Unknown")
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("Unknown/Invalid", "Unknown")

    # A1Cresult y max_glu_serum: rellenar NaN con "None" (categoria válida)
    if "A1Cresult" in df.columns:
        df["A1Cresult"] = df["A1Cresult"].fillna("None")
    if "max_glu_serum" in df.columns:
        df["max_glu_serum"] = df["max_glu_serum"].fillna("None")

    # Crear target binario
    if "readmitted" in df.columns:
        df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
    else:
        st.warning("No se encontró la columna 'readmitted' en el dataset.")

    # quitar identificadores
    for c in ["encounter_id", "patient_nbr", "readmitted"]:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Sanitizar nombres de columnas (para compatibilidad XGBoost)
    old_cols = df.columns.tolist()
    new_cols = [sanitize_colname(c) for c in old_cols]
    # evitar duplicados
    from collections import Counter
    cnt = Counter(new_cols)
    for i, name in enumerate(new_cols):
        if cnt[name] > 1:
            occ = sum(1 for j in range(i) if new_cols[j] == name) + 1
            new_cols[i] = f"{name}_{occ}"
    col_map = dict(zip(old_cols, new_cols))
    df.rename(columns=col_map, inplace=True)

    # Separar target y features
    if "readmitted_binary" not in df.columns:
        raise ValueError("Column 'readmitted_binary' missing after preprocess.")
    y = df["readmitted_binary"].copy()
    X = df.drop(columns=["readmitted_binary"])

    # Identificar numéricas
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Manejo de categóricas -> get_dummies (rápido y reproducible aquí)
    if use_get_dummies and len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    return X, y, col_map

# ------------------------
# Entrenamiento XGBoost (con scale_pos_weight)
# ------------------------
@st.cache_resource
def train_xgboost(X_train, y_train, X_valid, y_valid, params=None):
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    default_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )
    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    # Algunos entornos no aceptan early_stopping_rounds en fit(), así que fallback
    try:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        model.fit(X_train, y_train)

    return model

# ------------------------
# Entrenamiento RandomForest
# ------------------------
@st.cache_resource
def train_rf(X_train, y_train, n_estimators=200):
    model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model

# ------------------------
# Funciones métricas y optimización de umbral
# ------------------------
def optimize_thresholds(y_true, y_proba, min_prec=0.20):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx_best_f1 = np.nanargmax(f1_scores)
    best_threshold_f1 = thresholds[idx_best_f1]
    best_f1 = f1_scores[idx_best_f1]
    best_prec_f1 = precisions[idx_best_f1]
    best_rec_f1 = recalls[idx_best_f1]

    # ROC Youden
    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
    youden = tpr - fpr
    idx_best_youden = np.argmax(youden)
    best_threshold_youden = roc_thresh[idx_best_youden]

    # max recall con precision>=min_prec
    valid_idxs = np.where(precisions[:-1] >= min_prec)[0]
    best_threshold_prec_constraint = None
    best_prec = None
    best_rec = None
    if valid_idxs.size > 0:
        idx_choice = valid_idxs[np.argmax(recalls[:-1][valid_idxs])]
        best_threshold_prec_constraint = thresholds[idx_choice]
        best_prec = precisions[idx_choice]
        best_rec = recalls[idx_choice]

    return {
        "best_threshold_f1": best_threshold_f1,
        "best_f1": best_f1,
        "best_prec_f1": best_prec_f1,
        "best_rec_f1": best_rec_f1,
        "best_threshold_youden": best_threshold_youden,
        "best_threshold_prec_constraint": best_threshold_prec_constraint,
        "prec_constraint_prec": best_prec,
        "prec_constraint_rec": best_rec
    }

def eval_with_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm, y_pred

# ------------------------
# SHAP utils (matplotlib-safe)
# ------------------------
def shap_global_bar_plot(shap_values, X_valid, topk=20):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_names = X_valid.columns
    df_shap = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": shap_values.mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x="mean_abs_shap", y="feature", data=df_shap.head(topk), palette="pastel", ax=ax)
    ax.set_title(f"Top {topk} features por mean(|SHAP|)")
    plt.tight_layout()
    return fig, df_shap

def shap_local_plot(shap_values, X_valid, idx=0, topk=20):
    feat_names = X_valid.columns
    shap_vals_patient = shap_values[idx]
    feat_contribs = pd.DataFrame({
        "feature": feat_names,
        "shap_value": shap_vals_patient,
        "feature_value": X_valid.iloc[idx].values
    }).sort_values("shap_value", key=lambda s: np.abs(s), ascending=False).head(topk)
    fig, ax = plt.subplots(figsize=(8,6))
    colors = ["#ff7f7f" if v>0 else "#77c0b7" for v in feat_contribs["shap_value"]]
    ax.barh(feat_contribs["feature"][::-1], feat_contribs["shap_value"][::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value (impacto en la predicción: + => mayor riesgo)")
    ax.set_title(f"Contribuciones SHAP (paciente idx={idx}) — top {topk}")
    plt.tight_layout()
    return fig, feat_contribs

# ------------------------
# UI: Sidebar con la nueva pestaña "Descripción del dataset"
# ------------------------
page = st.sidebar.radio("Navegación", [
    "Cargar datos",
    "Descripción del dataset",
    "EDA",
    "Preprocesamiento",
    "Regresión Logística",
    "Árbol de Decisión",
    "Random Forest",   
    "Búsqueda de Umbral",
    "EXPLICABILIDAD",
    "Resultados / Export"
])




# ------------------------
# 1) Objetivos del proyecto
# ------------------------
    
if page == "Objetivos del proyecto":
    st.header("MÉTODOS DE MACHINE LEARNING EN LA BIOESTADÍSTICA")
    
    st.markdown("### Objetivo general")
    st.markdown("""
    Desarrollar modelos de machine learning que predigan el riesgo de resultados adversos (p. ej., reingreso hospitalario) en pacientes con diabetes mellitus, utilizando como variable principal el resultado de la HbA1c, e identificar factores asociados para generar explicaciones interpretables y útiles para la práctica clínica.
    """)
    st.markdown("### Pregunta de investigación")
    st.markdown("""
    ¿Cuál es el modelo que mejor se ajusta a la redicción de riesgo de resultados adversos en pacientes con diabetes mellitus?
    """)
    st.markdown("### Activiades a desarrollar")
    st.markdown("""
    - Realizar un análisis descriptivo de las características demográficas, clínicas y de tratamiento de los pacientes diabéticos hospitalizados.
    - Preprocesar los datos mediante limpieza, imputación y codificación adecuada para su uso en modelos predictivos.
    - Entrenar y evaluar modelos supervisados (regresión logística) para predecir reingreso hospitalario, incorporando HbA1c y variables relevantes.
    - Comparar desempeño de los modelos mediante métricas de discriminación y calibración.
    - Identificar las variables más influyentes mediante interpretabilidad global (importancias, SHAP) y local, para guiar decisiones clínicas.
    """)
# ------------------------
# 2) Cargar datos
# ------------------------
if page == "Cargar datos":
    st.header("Carga de datos")
    st.markdown("Sube el archivo `diabetic_data.csv` (o el ZIP ya descomprimido).")
    uploaded = st.file_uploader("Subir CSV", type=["csv"])
    if uploaded:
        df_raw = load_csv_from_file(uploaded)
        st.success(f"Archivo cargado: {uploaded.name}")
        st.write("Vista rápida:")
        st.dataframe(df_raw.head())
        st.write("Dimensiones:", df_raw.shape)
        # cache it in session state
        st.session_state["df_raw"] = df_raw
    else:
        st.info("También puedes indicar la ruta local del CSV si corres localmente.")
        if "df_raw" in st.session_state:
            st.write("Dataset ya cargado en la sesión.")
        else:
            st.warning("Aún no hay datos cargados.")

# ------------------------
# 3) Descripción del dataset
# ------------------------
if page == "Descripción del dataset":
    st.header("Descripción del dataset")

    if "df_raw" not in st.session_state:
        st.warning("Primero sube el dataset en la pestaña 'Cargar datos'.")
    else:
        df = st.session_state["df_raw"]
        st.markdown("### Importancia global de la diabetes mellitus")
        st.markdown("""
        La diabetes mellitus representa una de las crisis sanitarias más urgentes a nivel mundial. Según la última edición del IDF Diabetes Atlas (2025), aproximadamente 589 millones de adultos entre 20 y 79 años viven con diabetes, cifra que podría aumentar hasta 853 millones para 2050 si no se adoptan medidas efectivas.

        - En 2024, la diabetes fue responsable de 3,4 millones de muertes y generó un gasto sanitario global estimado en 1 billón de dólares estadounidenses.
        - La carga recae de manera desproporcionada sobre los países de ingresos bajos y medios, que concentran aproximadamente el 81 % de los adultos con diabetes, con una proporción significativa de casos no diagnosticados

        - Esta enfermedad crónica conlleva complicaciones graves como daño vascular, renal, ocular y aumento de mortalidad precoz, lo que resalta la necesidad imperiosa de mejorar la detección temprana, el acceso al tratamiento y las políticas de salud pública.
        """)
        st.markdown("### Importancia global de la diabetes mellitus en Colombia")
        st.markdown("""
        En Colombia, la situación también es preocupante. Datos del IDF Diabetes Atlas indican que el 8,4 % de la población adulta padece diabetes, lo que equivale a unos 3 033 800 casos en un total de 36 728 500 adultos. 
        En Bogotá, un estudio transversal realizado entre 2022 y 2023 muestra que el 11 % de los adultos tienen diabetes tipo 2, cifra superior a la estimada previamente, con elevadas tasas asociadas a factores como edad avanzada, obesidad abdominal, dislipidemia y bajo nivel educativo.
        Estos datos enfatizan la urgencia de intervenciones conjuntas entre políticas públicas, atención primaria y educación comunitaria para prevenir un aumento mayor de la prevalencia y sus complicaciones en el país.
        """)
        st.markdown("### Contexto y motivación")
        st.markdown("""
        La diabetes mellitus es una de las principales causas de morbilidad y mortalidad en el mundo. La medición de la hemoglobina glicosilada (HbA1c) es un indicador clave del control glucémico y se ha asociado con mejores resultados clínicos y menor riesgo de complicaciones. En el contexto hospitalario, la identificación y manejo oportuno del control glucémico representan una oportunidad para reducir reingresos y optimizar la atención. Nuestro proyecto se centra en analizar datos clínicos de hospitalizaciones en pacientes con diagnóstico de diabetes, explorando el papel de la HbA1c y otras variables en la predicción de riesgo.
        """)
        st.markdown("### Variables clave (breve diccionario)")
        st.markdown("""
        - **encounter_id, patient_nbr**: identificadores (se eliminarán en el preprocesamiento).
        - **race**: raza (Caucasian, AfricanAmerican, Asian, Hispanic, Other, ? -> Unknown).
        - **gender**: género (Male, Female, Unknown/Invalid).
        - **age**: rango de edad en intervalos de 10 años (ej. [50-60)).
        - **admission_type_id / discharge_disposition_id / admission_source_id**: información de admisión y alta.
        - **time_in_hospital**: días de estancia (1–14).
        - **num_lab_procedures, num_procedures, num_medications**: conteos de pruebas, procedimientos y medicamentos.
        - **number_outpatient / number_emergency / number_inpatient**: visitas previas (último año).
        - **max_glu_serum**: glucosa sérica categórica (>300, >200, normal, None).
        - **A1Cresult**: resultado HbA1c categórico (>8, >7, normal, None).
        - **change, diabetesMed**: cambios en medicación / si se prescribió medicación antidiabética.
        - **medicamentos individuales**: columnas binarias indicando uso (metformin, insulin, etc.).
        - **readmitted**: variable objetivo ("<30", ">30", "NO").
        """)
        st.markdown("### Resumen general")
        st.write(f"**Filas:** {df.shape[0]} — **Columnas:** {df.shape[1]}")
        st.write("**Tipos de variables:**")
        dtypes = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index().rename(columns={"index":"variable"})
        st.dataframe(dtypes)
        st.markdown("_Nota: 'None' en A1Cresult/max_glu_serum indica que la prueba no fue realizada (se conserva como categoría). ' ? ' puede indicar missing en algunas columnas._")

        st.markdown("### Valores faltantes (top 30)")
        miss = df.isnull().sum().sort_values(ascending=False)
        st.dataframe(miss[miss > 0].head(30).rename("n_missing").to_frame())

        st.markdown("### Distribución de la variable objetivo y variables relacionadas")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Conteo `readmitted`**")
            fig, ax = plt.subplots()
            df['readmitted'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel("Categoría")
            ax.set_ylabel("Cuenta")
            st.pyplot(fig)
        with col2:
            st.write("**`A1Cresult` (conteo)**")
            fig, ax = plt.subplots()
            if "A1Cresult" in df.columns:
                df['A1Cresult'].fillna("None").value_counts().plot(kind='bar', ax=ax)
            else:
                ax.text(0.2, 0.5, "No existe columna 'A1Cresult' en este dataset", fontsize=12)
            st.pyplot(fig)

        st.markdown("### Primeras filas y resumen numérico")
        st.write("Vista rápida (primeras 10 filas):")
        st.dataframe(df.head(10))
        st.write("Resumen estadístico de variables numéricas:")
        st.dataframe(df.describe().T)

        st.markdown("### Observaciones / recomendaciones para preprocesamiento")
        st.markdown("""
        - Mantener explícita la categoría `'None'` en `A1Cresult` y `max_glu_serum` (significa 'no medido').  
        - Reemplazar `?` por `'Unknown'` o `NaN` dependiendo del contexto (ej. `race`).  
        - Eliminar columnas con porcentaje extremadamente alto de missing (por ejemplo `weight` si >90%).  
        - Codificar variables categóricas (One-Hot) y escalar numéricas antes de modelar.  
        - Conservar `readmitted` para generar `readmitted_binary` (1 si `<30`, 0 otro).
        """)

# ------------------------
# 4) EDA
# ------------------------
if page == "EDA":
    st.header("Análisis Exploratorio de Datos (EDA)")
    if "df_raw" not in st.session_state:
        st.warning("Primero sube/indica el dataset en 'Cargar datos'.")
    else:
        df = st.session_state["df_raw"]
        st.subheader("Resumen")
        st.write(df.describe(include="all").T)
        st.subheader("Valores faltantes (top 20)")
        miss = df.isnull().sum().sort_values(ascending=False)
        st.write(miss[miss>0].head(20))
        st.subheader("Distribución variable objetivo `readmitted`")
        fig, ax = plt.subplots()
        df['readmitted'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("Categoria")
        ax.set_ylabel("Cuenta")
        st.pyplot(fig)

        st.subheader("Edad, género y raza")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots()
            df['age'].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_title("Edad")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.countplot(x="gender", data=df, ax=ax)
            ax.set_title("Género")
            st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots(figsize=(6,3))
            sns.countplot(x="race", data=df, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title("Raza")
            st.pyplot(fig)
        st.markdown("### ANÁLISIS")
        st.markdown("""
         + La mayoría de los pacientes están en los rangos [60–70), [70–80) y [50–60).

         + Muy pocos pacientes son jóvenes: casi no hay en [0–30).

        + Esto tiene sentido clínico: la diabetes es más prevalente en adultos mayores, y los riesgos de complicaciones aumentan con la edad.

        *Distribución por género:*
        + Mujeres (Female): ligeramente más que los hombres.

        + Hombres (Male): segunda categoría más frecuente.

        + Hay unos pocos casos marcados como Unknown/Invalid que probablemente debamos limpiar o recodificar.

        Esto refleja que la diabetes afecta tanto a hombres como a mujeres, sin un sesgo extremo de género en este dataset.

        *Distribución por raza:*

        + La gran mayoría de pacientes registrados son Caucasian.

        + El segundo grupo más frecuente es AfricanAmerican.

        + Minorías: Hispanic, Asian, Other, con números pequeños.

        Hay algunos valores "?", que representan missing values codificados como texto → estos debemos tratarlos en preprocesamiento (por ejemplo, recodificar como "Unknown").

        *Esto significa que, aunque el dataset es grande, puede no ser representativo de todas las poblaciones, algo a tener en cuenta cuando se interpreten los modelos.*
        """)

        st.subheader("Resultados de pruebas: HbA1c y glucosa sérica")

        # Asegurarse de usar la copia y rellenar NaN con 'None' (mantener la categoría)
        df_plot = df.copy()
        if "A1Cresult" in df_plot.columns:
            df_plot["A1Cresult"] = df_plot["A1Cresult"].fillna("None")
        else:
            st.warning("No existe la columna 'A1Cresult' en el dataset.")

        if "max_glu_serum" in df_plot.columns:
            df_plot["max_glu_serum"] = df_plot["max_glu_serum"].fillna("None")
        else:
            st.warning("No existe la columna 'max_glu_serum' en el dataset.")

        # Opcional: definir orden lógico de categorías si se conocen
        order_a1c = ["None", "Norm", ">7", ">8"]  # ajusta si tus categorías son distintas
        order_glu = ["None", "Norm", ">200", ">300"]  # ajusta según tus valores

        col1, col2 = st.columns(2)

        # --- HbA1c plot ---
        with col1:
            if "A1Cresult" in df_plot.columns:
                # preparar conteos en orden deseado (si una categoría falta, la agregamos con 0)
                vc = df_plot["A1Cresult"].value_counts()
                # si las categorías esperadas están en los datos, reindex; si no, usa el orden real
                cats = [c for c in order_a1c if c in vc.index] if any(c in vc.index for c in order_a1c) else list(vc.index)
                counts = vc.reindex(cats).fillna(0).astype(int)

                fig, ax = plt.subplots(figsize=(6,4))
                sns.barplot(x=counts.index, y=counts.values, palette="pastel", ax=ax)
                ax.set_title("Resultados de HbA1c (A1Cresult)")
                ax.set_xlabel("")
                ax.set_ylabel("Cuenta")

                # Anotar con conteos y porcentajes
                total = counts.sum()
                for i, v in enumerate(counts.values):
                    pct = (v / total * 100) if total>0 else 0
                    ax.text(i, v + max(counts.values)*0.01, f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

                st.pyplot(fig)

        # --- Glucosa sérica plot ---
        with col2:
            if "max_glu_serum" in df_plot.columns:
                vc2 = df_plot["max_glu_serum"].value_counts()
                cats2 = [c for c in order_glu if c in vc2.index] if any(c in vc2.index for c in order_glu) else list(vc2.index)
                counts2 = vc2.reindex(cats2).fillna(0).astype(int)

                fig2, ax2 = plt.subplots(figsize=(6,4))
                sns.barplot(x=counts2.index, y=counts2.values, palette="pastel", ax=ax2)
                ax2.set_title("Resultados de glucosa sérica (max_glu_serum)")
                ax2.set_xlabel("")
                ax2.set_ylabel("Cuenta")

                total2 = counts2.sum()
                for i, v in enumerate(counts2.values):
                    pct = (v / total2 * 100) if total2>0 else 0
                    ax2.text(i, v + max(counts2.values)*0.01, f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

                st.pyplot(fig2)
 
        st.markdown("### ANÁLISIS")
        st.markdown("""
            *Resultados de HbA1c*

            Categorías principales observadas:

            + >8 → la más frecuente, indica mal control glucémico.

            + Norm (normal) → cantidad intermedia.

            + >7 → menor frecuencia que las anteriores.

            La mayoría de pacientes que sí tienen un valor registrado de HbA1c presentan niveles altos (>8), lo que confirma que el dataset refleja una población de alto riesgo (pacientes hospitalizados con diabetes).
        
            *Resultados de glucosa sérica*

            Categorías principales:

            + Norm (normal) → es la más frecuente dentro de quienes tienen el dato.
            + >200 y >300 → representan casos de hiperglucemia importante, con menor frecuencia pero aún relevantes.
            La glucosa sérica muestra menos registros que HbA1c (muchos faltantes). Dentro de los que sí se midieron, predominan los normales, pero hay un grupo relevante con valores críticos (>200, >300).
        """)

        st.subheader("Distribuciones de variables numéricas")

        col3, col4 = st.columns(2)

        with col3:
            if "time_in_hospital" in df.columns:
                fig3, ax3 = plt.subplots(figsize=(6,4))
                sns.histplot(df['time_in_hospital'], bins=14, kde=False, color="#99ff99", ax=ax3)
                ax3.set_title("Distribución de días de hospitalización")
                ax3.set_xlabel("Días")
                ax3.set_ylabel("Frecuencia")
                # anotar total
                total = len(df['time_in_hospital'].dropna())
                ax3.text(0.95, 0.95, f"n={total}", transform=ax3.transAxes,
                         ha="right", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.6))
                st.pyplot(fig3)
            else:
                st.warning("No existe la columna 'time_in_hospital' en el dataset.")

        with col4:
            if "num_medications" in df.columns:
                fig4, ax4 = plt.subplots(figsize=(6,4))
                sns.histplot(df['num_medications'], bins=30, kde=False, color="#ffcc99", ax=ax4)
                ax4.set_title("Número de medicamentos administrados")
                ax4.set_xlabel("Medicamentos")
                ax4.set_ylabel("Frecuencia")
                total2 = len(df['num_medications'].dropna())
                ax4.text(0.95, 0.95, f"n={total2}", transform=ax4.transAxes,
                         ha="right", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.6))
                st.pyplot(fig4)
            else:
                st.warning("No existe la columna 'num_medications' en el dataset.")
        st.markdown("""
        *Distribución de días de hospitalización*

        + La mayoría de los pacientes estuvo hospitalizado entre 2 y 4 días.

        + A medida que aumentan los días, la frecuencia cae rápidamente.

        + El máximo permitido en el dataset es 14 días, pero muy pocos llegan a ese límite.

        + Los pacientes diabéticos suelen tener estancias cortas a moderadas, aunque los más graves pueden extenderse. Esto puede estar relacionado con el riesgo de reingreso.

        *Número de medicamentos administrados*

        + La mayoría recibió entre 5 y 20 medicamentos diferentes durante la hospitalización.

        + Hay una larga “cola” de pacientes que recibieron muchos más medicamentos (incluso >50), pero son casos poco frecuentes.

        + El número de medicamentos puede reflejar la complejidad clínica del paciente. Aquellos con polifarmacia (muchos medicamentos) probablemente tengan mayor riesgo de complicaciones y reingresos.
        """)
       
        st.subheader("Interpretación rápida de las distribuciones")

        # Asegurarse de que existen columnas relevantes
        has_med = "num_medications" in df.columns
        has_read = "readmitted" in df.columns
        has_a1c = "A1Cresult" in df.columns

        # Calcular medianas del número de medicamentos por categoría de readmitted
        if has_med and has_read:
            medians = df.groupby("readmitted")["num_medications"].median().reindex(["<30", ">30", "NO"])
            med_text = (
                f"Medianas de número de medicamentos (num_medications):\n"
                f"- Reingreso <30 días: {int(medians['<30']) if not pd.isna(medians['<30']) else 'N/A'}\n"
                f"- Reingreso >30 días: {int(medians['>30']) if not pd.isna(medians['>30']) else 'N/A'}\n"
                f"- No reingreso: {int(medians['NO']) if not pd.isna(medians['NO']) else 'N/A'}"
            )
        else:
            med_text = "No es posible calcular medianas de 'num_medications' por falta de columna(s)."

        # Calcular proporción de A1C >8 por readmitted (si existe)
        if has_a1c and has_read:
            df_a1c = df.copy()
            df_a1c["A1Cresult"] = df_a1c["A1Cresult"].fillna("None")
            pct_a1c = (df_a1c[df_a1c["A1Cresult"] == ">8"].groupby("readmitted").size() /
                       df_a1c.groupby("readmitted").size()).reindex(["<30", ">30", "NO"]) * 100
            pct_text = (
                f"Porcentaje de pacientes con A1C >8:\n"
                f"- Reingreso <30 días: {pct_a1c['<30']:.1f}%\n"
                f"- Reingreso >30 días: {pct_a1c['>30']:.1f}%\n"
                f"- No reingreso: {pct_a1c['NO']:.1f}%"
            )
        else:
            pct_text = "No es posible calcular proporciones de A1C >8 por falta de columna 'A1Cresult' o 'readmitted'."

        # Mostrar resultados numéricos (opcional)
        with st.expander("Ver estadísticas detalladas utilizadas para la interpretación"):
            st.markdown("**Medianas (num_medications) por categoría de readmitted:**")
            if has_med and has_read:
                st.write(medians)
            else:
                st.write("No disponible.")
            st.markdown("**Porcentaje de A1C >8 por categoría (si aplica):**")
            if has_a1c and has_read:
                st.write(pct_a1c)
            else:
                st.write("No disponible.")

        # Mostrar los textos interpretativos solicitados, con formato
        st.markdown("**Observaciones importantes**")
        st.markdown("""
        - **Los pacientes readmitidos (>30 y <30) tienden a recibir una mayor cantidad de medicamentos** en comparación con los que no fueron readmitidos.  
    
        - **La mediana del número de medicamentos es ligeramente mayor para aquellos con readmisión dentro de 30 días.**

        *Esto indica que tanto un mal control de glucosa (HbA1c alta) como el mayor número de medicamentos recetados están relacionados con una alta tasa de readmisión.*  
        """)

        # Añadir comentario numérico sobre HbA1c si se calculó
        if has_a1c and has_read:
            st.markdown("---")
            st.markdown("**Nota sobre HbA1c (>8):**")
            st.markdown(f"{pct_text}\n\nEstos porcentajes muestran que la fracción de pacientes con A1C >8 es mayor en las categorías de reingreso, sugiriendo asociación entre mal control glucémico y reingreso.")

# ------------------------
# 5) Preprocesamiento
# ------------------------
if page == "Preprocesamiento":
    st.header("Preprocesamiento")

    st.markdown("""
    El conjunto de datos original presentaba retos importantes de calidad y heterogeneidad que debieron abordarse antes de la modelación. En primer lugar, se identificaron valores faltantes y categorías especiales como “None”, que en este contexto no corresponden a datos perdidos sino a la indicación de que una prueba no fue realizada (por ejemplo, en las variables A1Cresult y max_glu_serum). Estas categorías se conservaron explícitamente como niveles válidos, permitiendo al modelo aprender del hecho de que una medición no haya sido solicitada. Por otro lado, los valores codificados como “?” en variables diagnósticas fueron tratados como ausentes y adecuadamente imputados o recategorizados. Posteriormente, las variables categóricas fueron transformadas mediante codificación One-Hot, mientras que las variables numéricas se normalizaron para garantizar escalas comparables entre predictores. Finalmente, dada la marcada desproporción entre clases (pacientes reingresados vs. no reingresados), se implementaron técnicas de balanceo de clases (SMOTE y el parámetro scale_pos_weight en XGBoost), con el fin de mitigar el sesgo hacia la clase mayoritaria y mejorar la capacidad de detección de reingresos.
    """)

    st.markdown("""
    **Transformaciones principales que se aplicarán:**
    - Eliminar columnas con altísimo porcentaje de missing (weight, payer_code, medical_specialty).
    - Reemplazar `?` en `race` por 'Unknown' y 'Unknown/Invalid' en `gender` por 'Unknown'.
    - Rellenar NaN de `A1Cresult` y `max_glu_serum` con 'None' (categoría válida).
    - Crear target binario `readmitted_binary` (1 si `<30`, 0 en otro caso).
    - Eliminar identificadores (`encounter_id`, `patient_nbr`, `readmitted`).
    - Sanitizar nombres de columnas.
    - One-Hot Encoding para variables categóricas.
    """)

    

    # Comprobar si el dataset fue cargado
    if "df_raw" not in st.session_state:
        st.warning("Sube el dataset primero en la pestaña 'Cargar datos'.")
    else:
        df = st.session_state["df_raw"]

        if st.button("Ejecutar preprocesamiento"):
            with st.spinner("Preprocesando..."):
                X, y, col_map = preprocess(df)
                st.session_state["X"] = X
                st.session_state["y"] = y
                st.session_state["col_map"] = col_map
            st.success("Preprocesamiento completado.")
            st.write("Dimensiones de las features:", X.shape)
            st.write("Balance de la variable objetivo (y):")
            st.write(y.value_counts())

# ------------------------
# Sección: Regresión Logística
# ------------------------
if page == "Regresión Logística":
    st.header("Resultados: Regresión Logística")

    # Mostrar métricas ya obtenidas (puedes reemplazar por las tuyas reales)
    st.subheader("Métricas de validación")
    st.markdown("""
    - **ROC-AUC:** 0.58  
    - **Precisión (clase 1):** 0.16  
    - **Recall (clase 1):** 0.42  
    - **F1-score (clase 1):** 0.23  
    """)

    # Matriz de confusión (imagen precargada o simulada)
    st.subheader("Matriz de confusión")
    fig, ax = plt.subplots()
    cm = [[11373, 2188],
          [1155, 549]]  # <- puedes reemplazar por tu matriz real
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Reingreso", "Reingreso<30"],
                yticklabels=["No Reingreso", "Reingreso<30"],
                ax=ax)
    ax.set_title("Matriz de confusión - Regresión Logística")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    st.pyplot(fig)

    # Curva ROC (usando puntos precalculados o un PNG exportado)
    st.subheader("Curva ROC")
    fig2, ax2 = plt.subplots()
    # Datos de ejemplo: reemplázalos por los de tu corrida
    fpr = [0.0, 0.2, 0.4, 0.6, 1.0]
    tpr = [0.0, 0.4, 0.6, 0.7, 1.0]
    ax2.plot(fpr, tpr, label="AUC = 0.58")
    ax2.plot([0,1], [0,1], "--", color="gray")
    ax2.set_xlabel("Tasa de falsos positivos")
    ax2.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax2.set_title("Curva ROC - Regresión Logística")
    ax2.legend()
    st.pyplot(fig2)

    # Conclusiones
    st.subheader("Interpretación")
    st.markdown("""
    + El modelo aprendió a predecir mayoritariamente la **clase 0** (no reingreso).
    + Aunque aplicamos **SMOTE en entrenamiento**, en validación (donde el desbalance se mantiene) el modelo **no generaliza bien**.
    + Esto confirma que una **regresión logística no es suficiente** para este problema:
        - No capta relaciones no lineales.
        - Hay muchas variables categóricas complejas que no aprovecha del todo.

    👉 Por ello se comparó con modelos **no lineales** (como Random Forest y XGBoost), que suelen funcionar mejor en datos tabulares y desbalanceados.
    """)

# ------------------------
# Árbol de Decisión 
# ------------------------
if page == "Árbol de Decisión":
    st.header("Resultados: Árbol de Decisión")

    # ====== MÉTRICAS RESUMEN (reemplaza por tus valores reales) ======
    auc_val = 0.59
    prec_cls1 = 0.18
    rec_cls1 = 0.28
    f1_cls1 = 0.22

    st.subheader("Métricas de validación")
    st.markdown(f"""
    - **ROC-AUC:** {auc_val:.2f}  
    - **Precisión (clase 1):** {prec_cls1:.2f}  
    - **Recall (clase 1):** {rec_cls1:.2f}  
    - **F1-score (clase 1):** {f1_cls1:.2f}  
    """)

    # ====== MATRIZ DE CONFUSIÓN (reemplaza por tu matriz real) ======
    st.subheader("Matriz de confusión")
    # Ejemplo de matriz (TN, FP / FN, TP). Cambia estos números por los tuyos.
    cm_tree = [
        [11200, 2361],   # [TN, FP]
        [1225,   479]    # [FN, TP]
    ]
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Reingreso", "Reingreso<30"],
                yticklabels=["No Reingreso", "Reingreso<30"],
                ax=ax_cm)
    ax_cm.set_title("Matriz de confusión - Árbol de Decisión")
    ax_cm.set_ylabel("Real")
    ax_cm.set_xlabel("Predicho")
    st.pyplot(fig_cm)

    # ====== CURVA ROC (reemplaza fpr/tpr por tus arrays reales si los tienes) ======
    st.subheader("Curva ROC")
    # Ejemplo de puntos ROC; si guardaste tus arrays reales, colócalos aquí.
    fpr_ex = [0.00, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00]
    tpr_ex = [0.00, 0.35, 0.48, 0.56, 0.62, 0.70, 1.00]
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_ex, tpr_ex, label=f"AUC = {auc_val:.2f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax_roc.set_title("Curva ROC - Árbol de Decisión")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # ====== IMPORTANCIA DE VARIABLES (reemplaza por tu ranking real si lo tienes) ======
    st.subheader("Importancia de variables (Top 15)")
    # Placeholder de importancias (nombre -> valor). Sustituye por las tuyas reales
    # Por ejemplo, si guardaste un CSV con 'feature' e 'importance', puedes leerlo y graficarlo.
    placeholder_importances = {
        "time_in_hospital": 0.085,
        "num_medications": 0.080,
        "num_lab_procedures": 0.072,
        "number_inpatient": 0.061,
        "age_[60-70)": 0.055,
        "A1Cresult_>8": 0.050,
        "insulin_Down": 0.046,
        "change_Ch": 0.041,
        "diabetesMed_Yes": 0.038,
        "max_glu_serum_>300": 0.036,
        "number_emergency": 0.033,
        "number_outpatient": 0.030,
        "admission_type_id": 0.028,
        "discharge_disposition_id": 0.025,
        "number_diagnoses": 0.024,
    }
    # Convertir a DataFrame y graficar
    importances_df = (
        pd.Series(placeholder_importances)
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
    )
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=importances_df, palette="pastel", ax=ax_imp)
    ax_imp.set_title("Top 15 variables más importantes - Árbol de Decisión")
    ax_imp.set_xlabel("Importancia")
    ax_imp.set_ylabel("")
    st.pyplot(fig_imp)

    # ====== INTERPRETACIÓN ======
    st.subheader("Interpretación")
    st.markdown("""
    + El **árbol** captura mejor la estructura del dataset que la **regresión logística**, pero **sigue sin ser excelente**.
    + El **recall** de la clase minoritaria (1) continúa **bajo** → el modelo todavía no detecta bien a los pacientes de **alto riesgo** (<30 días).
    + El **ROC-AUC ~0.59** indica una mejora respecto a la regresión, pero aún **lejos de un modelo robusto**.

    👉 Para este tipo de datos **tabulares y desbalanceados**, lo habitual es que los **modelos de ensamble** (**Random Forest** y **XGBoost**) rindan **mejor**.
    """)

    st.info("Tip: si ya tienes guardadas las salidas reales (fpr, tpr, cm, importances), "
            "puedo darte un snippet para leerlas desde CSV/JSON y mostrarlas aquí sin tocar el código.")


# ------------------------
# Random Forest 
# ------------------------
if page == "Random Forest":
    st.header("Resultados: Random Forest")

    # ====== MÉTRICAS RESUMEN (reemplaza por tus valores reales) ======
    auc_val = 0.64         # Ej.: tu ROC-AUC en validación/test
    prec_cls1 = 0.43       # Precisión de la clase 1 (ejemplo)
    rec_cls1  = 0.01       # Recall clase 1 (ejemplo)
    f1_cls1   = 0.02       # F1 clase 1 (ejemplo)

    st.subheader("Métricas de validación")
    st.markdown(f"""
    - **ROC-AUC:** {auc_val:.2f}  
    - **Precisión (clase 1):** {prec_cls1:.2f}  
    - **Recall (clase 1):** {rec_cls1:.2f}  
    - **F1-score (clase 1):** {f1_cls1:.2f}  
    """)

    # ====== MATRIZ DE CONFUSIÓN (reemplaza por tu matriz real) ======
    st.subheader("Matriz de confusión")
    # Ejemplo de matriz (TN, FP / FN, TP). Cambia por tus números reales.
    cm_rf = [
        [12050, 1511],   # [TN, FP]
        [1688,    16]    # [FN, TP]
    ]
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Reingreso", "Reingreso<30"],
                yticklabels=["No Reingreso", "Reingreso<30"],
                ax=ax_cm)
    ax_cm.set_title("Matriz de confusión - Random Forest")
    ax_cm.set_ylabel("Real")
    ax_cm.set_xlabel("Predicho")
    st.pyplot(fig_cm)

    # ====== CURVA ROC (reemplaza fpr/tpr por tus arrays reales si los tienes) ======
    st.subheader("Curva ROC")
    # Puntos de ejemplo; si guardaste tus arrays reales, colócalos aquí.
    fpr_ex = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 1.00]
    tpr_ex = [0.00, 0.10, 0.18, 0.30, 0.45, 0.58, 1.00]
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_ex, tpr_ex, label=f"AUC = {auc_val:.2f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax_roc.set_title("Curva ROC - Random Forest")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # ====== IMPORTANCIA DE VARIABLES (reemplaza por tu ranking real si lo tienes) ======
    st.subheader("Importancia de variables (Top 15)")
    # Placeholder: si guardaste tus importancias (feature_importances_), arma un dict {feature: importancia}
    importances_dict = {
        "time_in_hospital": 0.040,
        "num_medications": 0.038,
        "num_lab_procedures": 0.036,
        "number_inpatient": 0.032,
        "age_[60-70)": 0.030,
        "A1Cresult_>8": 0.028,
        "insulin_Down": 0.026,
        "change_Ch": 0.024,
        "diabetesMed_Yes": 0.023,
        "max_glu_serum_>300": 0.022,
        "number_emergency": 0.020,
        "number_outpatient": 0.019,
        "admission_type_id": 0.018,
        "discharge_disposition_id": 0.017,
        "number_diagnoses": 0.016,
    }
    importances_df = (
        pd.Series(importances_dict)
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
    )
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=importances_df, palette="pastel", ax=ax_imp)
    ax_imp.set_title("Top 15 variables más importantes - Random Forest")
    ax_imp.set_xlabel("Importancia")
    ax_imp.set_ylabel("")
    st.pyplot(fig_imp)

    # ====== INTERPRETACIÓN ======
    st.subheader("Interpretación")
    st.markdown("""
    + **Random Forest** mejoró el **ROC-AUC** global, pero **falló en captar la clase minoritaria**.
    + Esto pasa mucho en datasets clínicos **desbalanceados**: los modelos priorizan la **clase mayoritaria** para maximizar *accuracy*.
    + El **recall** de la **clase 1** (pacientes de riesgo) es **crítico** en este problema, y aquí quedó **prácticamente nulo**.

    👉 Por ello, se **ajusta el umbral de decisión** en el Random Forest para **mejorar el recall** de reingresos (ver pestaña **Umbral**).
    """)
# ------------------------
# Búsqueda de Umbral
# ------------------------
if page == "Búsqueda de Umbral":
    st.header("Búsqueda de umbral — Visualización estática (sin correr modelo)")

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # === Datos estáticos tomados de tus resultados ===
    data = pd.DataFrame({
        "threshold": [0.165, 0.195, 0.240],
        "precision": [0.159, 0.1724, 0.2010],
        "recall":    [0.6297, 0.4900, 0.3222],
        "f1":        [0.254, 0.2551, 0.2472],
        "accuracy":  [0.5865, 0.6805, 0.7810]
    })

    st.subheader("Tabla resumen (umbral vs métricas)")
    st.dataframe(data.style.format({
        "threshold": "{:.3f}",
        "precision": "{:.3f}",
        "recall": "{:.3f}",
        "f1": "{:.3f}",
        "accuracy": "{:.3f}"
    }), use_container_width=True)

    # === Gráfica: Precision / Recall / F1 vs Umbral ===
    st.subheader("Precision / Recall / F1 vs Umbral")
    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(data["threshold"], data["precision"], marker="o", label="Precisión")
    ax1.plot(data["threshold"], data["recall"], marker="o", label="Recall")
    ax1.plot(data["threshold"], data["f1"], marker="o", label="F1")
    ax1.set_xlabel("Umbral")
    ax1.set_ylabel("Valor")
    ax1.set_title("Métricas vs Umbral (valores reportados)")
    ax1.set_xticks(data["threshold"])
    ax1.grid(alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

    # === Gráfica: Accuracy vs Umbral ===
    st.subheader("Accuracy vs Umbral")
    fig2, ax2 = plt.subplots(figsize=(7,3.8))
    sns.barplot(x="threshold", y="accuracy", data=data, palette="pastel", ax=ax2)
    for i, row in data.iterrows():
        ax2.text(i, row["accuracy"] + 0.01, f"{row['accuracy']:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Umbral")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Exactitud por umbral")
    st.pyplot(fig2)

    # (Opcional) “radar” pequeño para ver trade-offs
    try:
        import math
        st.subheader("Comparación visual (radar) — Opcional")
        # Normalizamos métricas 0..1 para el radar
        radar_cols = ["precision", "recall", "f1", "accuracy"]
        angles = np.linspace(0, 2*np.pi, len(radar_cols), endpoint=False).tolist()
        angles += angles[:1]

        fig_r, ax_r = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        for idx, row in data.iterrows():
            values = [row[c] for c in radar_cols]
            values += values[:1]
            ax_r.plot(angles, values, marker="o", label=f"thr={row['threshold']:.3f}")
            ax_r.fill(angles, values, alpha=0.08)

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(radar_cols)
        ax_r.set_yticklabels([])
        ax_r.set_title("Radar: trade-off por umbral")
        ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
        st.pyplot(fig_r)
    except Exception:
        pass

    # === Conclusiones (texto tal cual lo pediste) ===
    st.subheader("Conclusiones e interpretación")

    st.markdown("""
    **Resumen de resultados con distintos umbrales**

    **Umbral ≈ 0.165 (muy bajo)**  
    Recall clase 1 (reingreso): 0.63 🚀 (detecta casi 2/3 de los casos en validación).  
    Precisión clase 1: 0.16 (muchos falsos positivos).  
    Accuracy: 0.58 (bajó bastante).  
    👉 Este escenario es muy sensible: atrapa la mayoría de pacientes en riesgo, pero con demasiadas alarmas falsas.

    **Umbral ≈ 0.240 (más conservador)**  
    Recall clase 1: 0.32 (detecta 1/3).  
    Precisión clase 1: 0.20 (un poco mejor).  
    Accuracy: 0.78 (más equilibrado).  
    👉 Buen balance si quieres algo intermedio: menos alarmas falsas, pero sacrificas detecciones.

    **Umbral ≈ 0.195 en TEST (generalización)**  
    Recall clase 1: 0.48 (casi la mitad de los reingresos detectados).  
    Precisión clase 1: 0.17.  
    Accuracy: 0.68.  
    👉 Este umbral parece un punto de compromiso realista: no tan extremo como 0.165, pero mucho más útil que el default 0.5 o que el árbol simple.
    """)

    st.markdown("""
    **Ajuste de umbral y desempeño del modelo**  
    Durante la evaluación se analizó el efecto de modificar el umbral de decisión en el modelo de predicción. Se observó que con el umbral estándar (0.5) el modelo privilegiaba la precisión de la clase mayoritaria, pero presentaba baja sensibilidad para detectar reingresos. Al reducir el umbral (0.165–0.195) se incrementó notablemente el recall de la clase positiva (hasta ~63%), permitiendo identificar más pacientes en riesgo, aunque a costa de una disminución de la exactitud global y un mayor número de falsos positivos. En cambio, al aumentar ligeramente el umbral (0.240) se logró un mayor equilibrio entre precisión y sensibilidad, con una exactitud global más alta (~78%), pero un menor recall en la clase de reingresos (~32%). En la evaluación final sobre el conjunto de prueba con umbral = 0.195, se obtuvo un recall de ~48% y un F1 de ~0.25 para la clase positiva, confirmando que la **selección del umbral depende del criterio clínico**.
    """)

    st.markdown("""
    **Resultados de la optimización de umbrales**  
    Se exploraron distintos criterios para seleccionar el umbral:  
    - **Mejor F1-score**: 0.195 → equilibrio relativo (Prec≈0.17, Rec≈0.49).  
    - **Youden (ROC)**: 0.165 → máxima sensibilidad pero mucha falsa alarma.  
    - **Máx. Recall con Precisión ≥ 0.20**: 0.240 → recall≈0.32 y desempeño más balanceado.  
    La **elección del umbral es clínica**, no solo técnica: depende de priorizar detección (recall) versus uso eficiente de recursos (precisión).
    """)

    st.markdown("""
    **Decisiones según el objetivo clínico**  
    + Si el objetivo principal es **detectar más reingresos** aunque haya falsos positivos → usar umbral **bajo (≈0.18–0.20)**.  
    + Si además interesa la **eficiencia hospitalaria** (reducir falsas alarmas) → usar umbral **≈0.25**.  

    *Para complementar el trabajo, se implementa **XGBoost** con ajuste de desbalance (`scale_pos_weight`) y se compara contra **Random Forest con umbral ajustado**.*
    """)




# ------------------------
# 8) EXPLICABILIDAD
# ------------------------
if page == EXPLICABILIDAD":
    st.header("Explicabilidad con SHAP (modo matplotlib / HTML)")
    if "model_xgb" not in st.session_state:
        st.warning("Entrena XGBoost primero (recomendado).")
    else:
        model = st.session_state["model_xgb"]
        X_valid = st.session_state["X_valid"]

        st.write("Computando valores SHAP (TreeExplainer). Esto puede tardar varios segundos/minutos según el tamaño del conjunto de validación.")
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_valid)  # puede ser array (n, m)

        st.subheader("Importancia global (matplotlib-safe)")
        fig_bar, df_shap = shap_global_bar_plot(shap_values, X_valid, topk=20)
        st.pyplot(fig_bar)
        st.write("Tabla top 20 features (mean|SHAP|):")
        st.dataframe(df_shap.head(20))

        st.subheader("Explicación local (paciente)")
        idx = st.number_input("Índice de paciente (en validación)", min_value=0, max_value=len(X_valid)-1, value=10, step=1)
        fig_local, feat_contribs = shap_local_plot(shap_values, X_valid, idx=int(idx), topk=20)
        st.pyplot(fig_local)
        st.dataframe(feat_contribs)

        st.markdown("### Force plot interactivo (opcional)")
        st.markdown("Si quieres el *force_plot* interactivo, podemos generar el HTML y embeberlo (a veces no funciona en Streamlit Cloud).")
        if st.button("Generar force_plot HTML (puede tardar)"):
            fp = shap.force_plot(explainer.expected_value, shap_values[int(idx), :], X_valid.iloc[int(idx), :])
            # Guardar HTML temporal
            shap.save_html("shap_force_plot.html", fp)
            # Leer HTML y mostrar con componente
            import streamlit.components.v1 as components
            with open("shap_force_plot.html", "r") as f:
                html_str = f.read()
            components.html(html_str, height=600, scrolling=True)

# ------------------------
# 9) Resultados / Export
# ------------------------
if page == "Resultados / Export":
    st.header("Resultados finales y exportación")
    st.write("Modelos entrenados en la sesión:")
    if "model_xgb" in st.session_state:
        st.write("- XGBoost disponible")
    if "model_rf" in st.session_state:
        st.write("- Random Forest disponible")

    if st.button("Guardar modelo XGBoost (joblib)"):
        if "model_xgb" in st.session_state:
            joblib.dump(st.session_state["model_xgb"], "model_xgb.joblib")
            st.success("Modelo XGBoost guardado: model_xgb.joblib")
        else:
            st.warning("No hay modelo XGBoost en memoria.")

    if st.button("Guardar X y col_map (pickles)"):
        if "X" in st.session_state and "col_map" in st.session_state:
            joblib.dump(st.session_state["X"], "X_features.joblib")
            joblib.dump(st.session_state["col_map"], "col_map.joblib")
            st.success("Guardados X_features.joblib y col_map.joblib")
        else:
            st.warning("No hay X / col_map en sesión.")

    st.markdown("### CONCLUSIONES")
    st.markdown("""
        
        *Interpretación desde el punto de vista médico*

        **Recall alto (sensibilidad alta):**

        + Detecta la mayoría de los pacientes que realmente se reingresarán.

        + Ventaja: menos falsos negativos (menos pacientes de riesgo que se escapan).

        + Desventaja: más falsos positivos → se etiquetan como “en riesgo” pacientes que no lo están → puede generar sobrecarga en programas de seguimiento, gasto extra de recursos o alarmas innecesarias.

        **Precisión alta (pocos falsos positivos):**

        + Cuando el modelo predice “riesgo de reingreso”, es bastante confiable.

        + Ventaja: se usan los recursos de seguimiento de manera más focalizada.

        + Desventaja: bajo recall → se dejan pasar muchos pacientes en riesgo real sin detectar.

        **Equilibrio (F1-score):**

        + Un término medio: ni se maximizan detecciones a costa de muchos falsos positivos, ni se minimizan recursos a costa de dejar pasar pacientes críticos.

        + Suele ser la opción más razonable en estudios iniciales, sobre todo si el hospital tiene recursos limitados pero quiere detectar una fracción significativa de casos de riesgo.

        **Interpretabilidad**

        “En este estudio se entrenaron varios modelos de machine learning para predecir reingreso hospitalario en pacientes diabéticos. El modelo con mejor desempeño fue XGBoost, alcanzando un AUC ≈0.67–0.69. Según el criterio clínico, si se prioriza la detección de la mayor cantidad de pacientes en riesgo (recall 56%), conviene mantener el umbral estándar (0.5). En cambio, si se busca un balance entre recall y precisión (F1=0.27), conviene ajustar el umbral a ≈0.57. La decisión final depende de los recursos disponibles y de la estrategia clínica que adopte la institución.”
    """)

    st.markdown("---")
    st.markdown("Carolina Rojas-Jorge Marulanda-Jheremy Reyes")

