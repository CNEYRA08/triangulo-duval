import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io

# --- CONFIGURACIÓN DE PÁGINA (ESTILO VOLTIUM) ---
st.set_page_config(page_title="Voltium | DGA Duval", layout="wide")

# Título y Branding
st.title("⚡ Simulador de Diagnóstico DGA - Triángulo de Duval")
st.markdown(
    "<div style='text-align:right; font-weight:bold; color:#00AEEF; letter-spacing:4px;'>VOLTIUM</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# --- LÓGICA MATEMÁTICA (Del archivo original + Adaptaciones) ---

def tern2cart(ch4, c2h4, c2h2):
    """
    Convierte porcentajes (CH4, C2H4, C2H2) a coordenadas cartesianas.
    Triángulo: CH4=100% en (0.5, 1), C2H4=100% en (1, 0), C2H2=100% en (0, 0).
    """
    ch4, c2h4, c2h2 = np.atleast_1d(ch4), np.atleast_1d(c2h4), np.atleast_1d(c2h2)
    x = c2h4 / 100 + ch4 / 200
    y = ch4 / 100
    return np.squeeze(x), np.squeeze(y)


def segmento_ternario(constante, tipo, valor):
    """
    Devuelve puntos (ch4, c2h4, c2h2) del segmento dentro del triángulo.
    tipo: 'CH4', 'C2H4' o 'C2H2'. Usado para dibujar líneas de referencia.
    """
    if tipo == "CH4":
        restante = 100 - valor
        if restante <= 0:
            return [], [], []
        c2h4 = np.linspace(0, restante, 50)
        c2h2 = restante - c2h4
        ch4 = np.full_like(c2h4, valor)
        return ch4, c2h4, c2h2
    if tipo == "C2H4":
        restante = 100 - valor
        if restante <= 0:
            return [], [], []
        ch4 = np.linspace(0, restante, 50)
        c2h2 = restante - ch4
        c2h4 = np.full_like(ch4, valor)
        return ch4, c2h4, c2h2
    if tipo == "C2H2":
        restante = 100 - valor
        if restante <= 0:
            return [], [], []
        ch4 = np.linspace(0, restante, 50)
        c2h4 = restante - ch4
        c2h2 = np.full_like(ch4, valor)
        return ch4, c2h4, c2h2
    return [], [], []


def clasificar_duval(ch4, c2h4, c2h2):
    """
    Clasifica según Tabla 6 / Figura 3 (Duval Triángulo 1).
    ch4, c2h4, c2h2: porcentajes (suma = 100). Orden de evaluación evita solapamientos.
    """
    total = ch4 + c2h4 + c2h2
    if total == 0:
        return "N/A"
    if abs(ch4 + c2h4 + c2h2 - 100) > 0.01:
        return "Fuera (suma ≠ 100%)"

    if ch4 >= 98:
        return "PD"
    if c2h2 < 4 and c2h4 < 20:
        return "T1"
    if c2h2 < 4 and 20 <= c2h4 < 50:
        return "T2"
    if c2h2 < 15 and c2h4 >= 50:
        return "T3"
    if c2h2 >= 13 and c2h4 < 23:
        return "D1"
    if c2h4 >= 23 and c2h2 >= 29:
        return "D2"
    if 23 <= c2h4 < 40 and 13 <= c2h2 < 29:
        return "D2"
    if 4 <= c2h2 < 13 and c2h4 < 50:
        return "DT"
    if 40 <= c2h4 < 50 and 13 <= c2h2 < 29:
        return "DT"
    if c2h4 >= 50 and 15 <= c2h2 < 29:
        return "DT"
    return "DT"

# --- TABLAS IEEE C57.104-2019 (P90 y P95) ---
# Ratio O2/N2: leq02 = ≤ 0.2 (Sellado), gt02 = > 0.2 (Respiración libre).
# Edad: desc = Desconocida, 1_9, 10_30, 30 = >30 años.
# Valores en µL/L (ppm). Celdas en blanco = mismo que el valor previo en el bloque.

GASES_IEEE = ["H2", "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2"]
GASES_LABELS = {
    "H2": "Hidrógeno (H₂)",
    "CH4": "Metano (CH₄)",
    "C2H6": "Etano (C₂H₆)",
    "C2H4": "Etileno (C₂H₄)",
    "C2H2": "Acetileno (C₂H₂)",
    "CO": "Monóxido de C. (CO)",
    "CO2": "Dióxido de C. (CO₂)",
}

TABLA_P90 = {
    "leq02": {
        "desc": {"H2": 80, "CH4": 90, "C2H6": 90, "C2H4": 50, "C2H2": 1, "CO": 900, "CO2": 9000},
        "1_9": {"H2": 80, "CH4": 45, "C2H6": 30, "C2H4": 20, "C2H2": 1, "CO": 900, "CO2": 5000},
        "10_30": {"H2": 75, "CH4": 90, "C2H6": 90, "C2H4": 50, "C2H2": 1, "CO": 900, "CO2": 10000},
        "30": {"H2": 100, "CH4": 110, "C2H6": 150, "C2H4": 90, "C2H2": 1, "CO": 900, "CO2": 5000},
    },
    "gt02": {
        "desc": {"H2": 40, "CH4": 20, "C2H6": 15, "C2H4": 50, "C2H2": 2, "CO": 500, "CO2": 3500},
        "1_9": {"H2": 40, "CH4": 20, "C2H6": 15, "C2H4": 25, "C2H2": 2, "CO": 500, "CO2": 3500},
        "10_30": {"H2": 40, "CH4": 20, "C2H6": 15, "C2H4": 60, "C2H2": 2, "CO": 500, "CO2": 5500},
        "30": {"H2": 40, "CH4": 20, "C2H6": 15, "C2H4": 60, "C2H2": 2, "CO": 500, "CO2": 5500},
    },
}

TABLA_P95 = {
    "leq02": {
        "desc": {"H2": 200, "CH4": 150, "C2H6": 175, "C2H4": 100, "C2H2": 2, "CO": 1100, "CO2": 12500},
        "1_9": {"H2": 200, "CH4": 100, "C2H6": 70, "C2H4": 40, "C2H2": 2, "CO": 1100, "CO2": 7000},
        "10_30": {"H2": 200, "CH4": 150, "C2H6": 175, "C2H4": 95, "C2H2": 2, "CO": 1100, "CO2": 14000},
        "30": {"H2": 200, "CH4": 200, "C2H6": 250, "C2H4": 175, "C2H2": 4, "CO": 1100, "CO2": 14000},
    },
    "gt02": {
        "desc": {"H2": 90, "CH4": 50, "C2H6": 40, "C2H4": 100, "C2H2": 7, "CO": 600, "CO2": 7000},
        "1_9": {"H2": 90, "CH4": 60, "C2H6": 30, "C2H4": 80, "C2H2": 7, "CO": 600, "CO2": 5000},
        "10_30": {"H2": 90, "CH4": 60, "C2H6": 40, "C2H4": 125, "C2H2": 7, "CO": 600, "CO2": 8000},
        "30": {"H2": 90, "CH4": 80, "C2H6": 40, "C2H4": 125, "C2H2": 7, "CO": 600, "CO2": 8000},
    },
}


def ieee_paso1_clasificar_sistema(o2_ppm, n2_ppm):
    """Paso 1: Ratio O2/N2. ≤ 0.2 Sellado, > 0.2 Respiración libre."""
    if n2_ppm is None or n2_ppm == 0:
        return None, None
    ratio = (o2_ppm or 0) / n2_ppm
    if ratio <= 0.2:
        return "leq02", "Sellado (O₂/N₂ ≤ 0.2)"
    return "gt02", "Respiración libre (O₂/N₂ > 0.2)"


def ieee_obtener_limites(ratio_key, age_key):
    """Devuelve límites P90 y P95 para (ratio, edad)."""
    p90 = TABLA_P90.get(ratio_key, {}).get(age_key, {})
    p95 = TABLA_P95.get(ratio_key, {}).get(age_key, {})
    return p90, p95


def ieee_paso3_condicion(valores_ppm, p90, p95):
    """
    Paso 3: Condición 1 (Normal), 2 (Precaución) o 3 (Alta/Alerta).
    valores_ppm: dict {gas: ppm} para GASES_IEEE.
    """
    alguno_p95 = False
    alguno_p90 = False
    for g in GASES_IEEE:
        v = valores_ppm.get(g) or 0
        lim95 = p95.get(g) or 0
        lim90 = p90.get(g) or 0
        if v >= lim95:
            alguno_p95 = True
        if v >= lim90:
            alguno_p90 = True
    if alguno_p95:
        return 3, "Condición 3 (Alta/Alerta)", "Los gases superan el percentil 95. Alta probabilidad de falla activa o reciente; se requiere investigación inmediata."
    if alguno_p90:
        return 2, "Condición 2 (Precaución)", "Al menos un gas supera el percentil 90 pero es menor al 95. Aumentar la frecuencia de muestreo para vigilar la tendencia."
    return 1, "Condición 1 (Normal)", "Todos los gases están por debajo del percentil 90. Continuar con el muestreo normal."


def get_fault_details(code):
    """Diccionario de interpretaciones técnicas."""
    details = {
        "PD": ("Descargas Parciales", "Descargas tipo corona, posibles vacíos en el aislamiento sólido o burbujas de gas en el aceite.", "Revisar nivel de aceite, buscar ruidos ultrasónicos."),
        "T1": ("Falla Térmica < 300°C", "Sobrecarga del papel o aceite, conexiones oxidadas pero de baja temperatura.", "Verificar historial de carga y estado del sistema de refrigeración."),
        "T2": ("Falla Térmica 300°C - 700°C", "Carbonización del papel, contactos defectuosos, corrientes circulantes en el núcleo.", "Inspección termográfica externa, planificar mantenimiento."),
        "T3": ("Falla Térmica > 700°C", "Puntos calientes severos, flujos de dispersión en tanque, cortocircuitos en el núcleo.", "Riesgo alto. Considerar desgasificación o inspección interna urgente."),
        "D1": ("Descargas de Baja Energía", "Chispas (sparking), descargas continuas de baja corriente.", "Pruebas eléctricas (resistencia de aislamiento, TTR) recomendadas."),
        "D2": ("Descargas de Alta Energía", "Arcos eléctricos severos (arcing), cortocircuitos francos entre espiras o a tierra.", "CRÍTICO. Sacar de servicio si la tasa de generación es alta. Análisis de furanos."),
        "DT": ("Falla Térmica y Eléctrica", "Mezcla de fallas. Posible arco con punto caliente.", "Investigación profunda requerida."),
        "N/A": ("Sin datos suficientes", "Ingrese valores mayores a 0", "-"),
        "Fuera (suma ≠ 100%)": ("Fuera del triángulo", "Los % de CH4, C2H4 y C2H2 deben sumar 100%.", "Verifique o normalice los valores."),
    }
    return details.get(code, ("Desconocido", "Zona no clasificada", "Revisar datos"))

# --- GENERADOR DE GRÁFICOS ---
def plot_duval_triangle(ch4_p, c2h4_p, c2h2_p, fault_code):
    """
    Genera el triángulo de Duval según Figura 3 y Tabla 6.
    Vértices: CH4 (arriba), C2H4 (abajo derecha), C2H2 (abajo izquierda).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Vértices: (0,0)=C2H2, (1,0)=C2H4, (0.5,1)=CH4
    verts = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])
    ax.plot(verts[:, 0], verts[:, 1], "k-", lw=2)

    # Límites de zonas (Tabla 6)
    limites_ch4 = [98]
    limites_c2h4 = [20, 23, 40, 50]
    limites_c2h2 = [4, 13, 15, 29]

    # Dibujar líneas de referencia
    for v in limites_ch4:
        ch4, c2h4, c2h2 = segmento_ternario(v, "CH4", v)
        if len(ch4):
            x, y = tern2cart(ch4, c2h4, c2h2)
            ax.plot(x, y, "k-", lw=1, alpha=0.8)
    for v in limites_c2h4:
        ch4, c2h4, c2h2 = segmento_ternario(v, "C2H4", v)
        if len(ch4):
            x, y = tern2cart(ch4, c2h4, c2h2)
            ax.plot(x, y, "k-", lw=1, alpha=0.8)
    for v in limites_c2h2:
        ch4, c2h4, c2h2 = segmento_ternario(v, "C2H2", v)
        if len(ch4):
            x, y = tern2cart(ch4, c2h4, c2h2)
            ax.plot(x, y, "k-", lw=1, alpha=0.8)

    # Malla para rellenar zonas (evitar 0 y 1 para estabilidad)
    n = 120
    xx = np.linspace(1e-6, 1 - 1e-6, n)
    yy = np.linspace(1e-6, 1 - 1e-6, n)
    X, Y = np.meshgrid(xx, yy)
    ch4_g = Y * 100
    c2h4_g = (X - Y / 2) * 100
    c2h2_g = 100 - ch4_g - c2h4_g

    inside = (ch4_g >= 0) & (c2h4_g >= 0) & (c2h2_g >= 0)
    Z = np.full_like(ch4_g, np.nan)
    zonas = ["PD", "T1", "T2", "T3", "D1", "D2", "DT"]
    colores = {
        "PD": "#FFE4B5",
        "T1": "#98FB98",
        "T2": "#90EE90",
        "T3": "#00FA9A",
        "D1": "#FFB6C1",
        "D2": "#FF69B4",
        "DT": "#DDA0DD",
    }
    c4, c24, c22 = ch4_g, c2h4_g, c2h2_g
    Z[inside] = 6
    Z[inside & (c4 >= 98)] = 0
    Z[inside & (c22 < 4) & (c24 < 20)] = 1
    Z[inside & (c22 < 4) & (c24 >= 20) & (c24 < 50)] = 2
    Z[inside & (c22 < 15) & (c24 >= 50)] = 3
    Z[inside & (c22 >= 13) & (c24 < 23)] = 4
    Z[inside & (c24 >= 23) & (c22 >= 29)] = 5
    Z[inside & (c24 >= 23) & (c24 < 40) & (c22 >= 13) & (c22 < 29)] = 5
    Z[inside & (c22 >= 4) & (c22 < 13) & (c24 < 50)] = 6
    Z[inside & (c24 >= 40) & (c24 < 50) & (c22 >= 13) & (c22 < 29)] = 6
    Z[inside & (c24 >= 50) & (c22 >= 15) & (c22 < 29)] = 6

    for idx, zona in enumerate(zonas):
        mask = (Z == idx) & inside
        if not np.any(mask):
            continue
        Z_zona = np.where(mask, 1.0, np.nan)
        ax.contourf(X, Y, Z_zona, levels=[0.5, 1.5], colors=[colores[zona]], alpha=0.6)

    # Etiquetas de zonas: puntos interiores en % (ch4, c2h4, c2h2) convertidos con tern2cart
    # para que queden sobre la región correspondiente
    etiquetas_tern = [
        (99, 0.5, 0.5, "PD"),   # PD: CH4 >= 98
        (87, 10, 3, "T1"),      # T1: C2H2<4, C2H4<20
        (60, 35, 2, "T2"),      # T2: C2H2<4, 20<=C2H4<50
        (25, 65, 10, "T3"),     # T3: C2H2<15, C2H4>=50
        (50, 15, 35, "D1"),     # D1: C2H2>=13, C2H4<23
        (25, 35, 40, "D2"),     # D2: C2H4>=23, C2H2>=13
        (40, 35, 25, "DT"),     # DT: zona central
    ]
    for ch4, c2h4, c2h2, texto in etiquetas_tern:
        x, y = tern2cart(ch4, c2h4, c2h2)
        ax.text(x, y, texto, fontsize=9, fontweight="bold", ha="center", va="center", color="#444")

    # Punto del usuario
    user_x, user_y = tern2cart(ch4_p, c2h4_p, c2h2_p)
    ax.plot(user_x, user_y, marker="*", markersize=18, color="red", markeredgecolor="black", label="Punto actual")

    # Etiquetas de ejes
    ax.text(0.5, 1.05, "% CH₄", fontsize=10, ha="center")
    ax.text(-0.06, -0.04, "% C₂H₂", fontsize=10, ha="center")
    ax.text(1.06, -0.04, "% C₂H₄", fontsize=10, ha="center")

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig

# --- INTERFAZ DE USUARIO ---

# Sidebar
st.sidebar.header("🎛️ Datos del Transformador")
st.sidebar.caption("_Aplica al Triángulo de Duval._")

# Opción de entrada
input_mode = st.sidebar.radio("Unidades de entrada:", ["PPM (Partes por millón)", "% Porcentaje Relativo"])

if input_mode == "PPM (Partes por millón)":
    st.sidebar.caption("Ingresa los valores del reporte DGA:")
    val_ch4 = st.sidebar.number_input("Metano (CH4)", min_value=0.0, value=10.0)
    val_c2h4 = st.sidebar.number_input("Etileno (C2H4)", min_value=0.0, value=10.0)
    val_c2h2 = st.sidebar.number_input("Acetileno (C2H2)", min_value=0.0, value=10.0)
    
    total = val_ch4 + val_c2h4 + val_c2h2
    if total > 0:
        pct_ch4 = (val_ch4 / total) * 100
        pct_c2h4 = (val_c2h4 / total) * 100
        pct_c2h2 = (val_c2h2 / total) * 100
    else:
        pct_ch4, pct_c2h4, pct_c2h2 = 0, 0, 0
else:
    st.sidebar.caption("Ingresa los porcentajes relativos (suma 100%):")
    pct_ch4 = st.sidebar.slider("% Metano (CH4)", 0.0, 100.0, 33.3)
    pct_c2h4 = st.sidebar.slider("% Etileno (C2H4)", 0.0, 100.0, 33.3)
    # Ajuste automático del tercero para guiar al usuario
    resto = 100.0 - pct_ch4 - pct_c2h4
    if resto < 0: resto = 0
    st.sidebar.info(f"El % Acetileno (C2H2) calculado es: {resto:.1f}%")
    pct_c2h2 = resto
    
    # Normalizar si el usuario mueve sliders locamente
    total_slider = pct_ch4 + pct_c2h4 + pct_c2h2
    if total_slider > 0 and abs(total_slider - 100) > 1:
        pct_ch4 = (pct_ch4 / total_slider) * 100
        pct_c2h4 = (pct_c2h4 / total_slider) * 100
        pct_c2h2 = (pct_c2h2 / total_slider) * 100

# Cálculo
diagnostico = clasificar_duval(pct_ch4, pct_c2h4, pct_c2h2)
nombre_falla, descripcion_falla, recomendacion = get_fault_details(diagnostico)

# --- PESTAÑAS PRINCIPALES ---
tab_duval, tab_ieee = st.tabs(["📐 Triángulo de Duval", "📋 Límites IEEE (P90/P95)"])

with tab_duval:
    # --- COLUMNAS PRINCIPALES (DUVAL) ---
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("🔍 Resultados del Análisis")
        m1, m2, m3 = st.columns(3)
        m1.metric("CH4 (Metano)", f"{pct_ch4:.1f}%")
        m2.metric("C2H4 (Etileno)", f"{pct_c2h4:.1f}%")
        m3.metric("C2H2 (Acetileno)", f"{pct_c2h2:.1f}%")
        st.divider()
        if diagnostico == "PD": color_res = "orange"
        elif diagnostico in ["T1", "T2", "T3"]: color_res = "#FF4B4B"
        elif diagnostico in ["D1", "D2", "DT"]: color_res = "#800080"
        else: color_res = "gray"
        st.markdown(f"""
        <div style="padding:20px; border-radius:10px; border:2px solid {color_res}; background-color: rgba(0,0,0,0.05);">
            <h2 style="color:{color_res}; margin:0;">Código: {diagnostico}</h2>
            <h3 style="margin-top:5px;">{nombre_falla}</h3>
            <p><strong>Causa Probable:</strong> {descripcion_falla}</p>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"💡 **Recomendación Voltium:** {recomendacion}")
        if "dga_history" not in st.session_state:
            st.session_state["dga_history"] = []
        id_trafo = st.text_input("Identificador del Transformador (Opcional)", "Trafo-01")
        if st.button("💾 Guardar en Historial de Sesión"):
            registro = {
                "ID": id_trafo,
                "CH4_pct": round(pct_ch4, 1),
                "C2H4_pct": round(pct_c2h4, 1),
                "C2H2_pct": round(pct_c2h2, 1),
                "Código": diagnostico,
                "Diagnóstico": nombre_falla
            }
            st.session_state["dga_history"].append(registro)
            st.success("Registro añadido.")

    with col2:
        st.subheader("📐 Gráfico de Duval")
        if pct_ch4 + pct_c2h4 + pct_c2h2 > 0:
            fig = plot_duval_triangle(pct_ch4, pct_c2h4, pct_c2h2, diagnostico)
            st.pyplot(fig)
        else:
            st.warning("Ingresa valores mayores a 0 para generar el gráfico.")

    st.markdown("---")
    st.subheader("📋 Historial de Análisis (Sesión Actual)")
    if len(st.session_state["dga_history"]) > 0:
        df_hist = pd.DataFrame(st.session_state["dga_history"])
        st.dataframe(df_hist, width="stretch")
        csv_buffer = io.StringIO()
        df_hist.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Descargar Reporte CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="reporte_dga_voltium.csv",
            mime="text/csv",
        )
    else:
        st.caption("Aún no hay registros guardados en esta sesión.")

    with st.expander("📚 Teoría: Triángulo de Duval 1 (IEEE C57.104 / IEC 60599)"):
        st.markdown("""
        El **Triángulo de Duval 1** se utiliza para gases generados por fallas en transformadores llenos de aceite mineral.
        Utiliza tres gases clave que corresponden al aumento de energía de la falla:

        * **CH4 (Metano):** Característico de puntos calientes de baja temperatura.
        * **C2H4 (Etileno):** Característico de puntos calientes de alta temperatura (aceite quemado).
        * **C2H2 (Acetileno):** Característico de arcos eléctricos (muy alta energía).

        **Nota:** Este método se debe aplicar solo cuando existe una sospecha de falla (niveles de gas por encima de los límites normales o incremento súbito de la tasa de generación).
        """)

with tab_ieee:
    st.subheader("Procedimiento de Diagnóstico DGA (IEEE Std C57.104-2019)")
    st.markdown("Determine si los valores de gas son *muy altos* antes de usar el Triángulo de Duval. Compare con las tablas de percentiles según sistema de preservación y edad.")

    with st.expander("📌 Criterios (Pasos 2 y 3)"):
        st.markdown("""
        **Percentil 90 (P90):** El 90% de los transformadores sanos de esa edad tienen menos gas que ese valor. Límite de Condición 1.

        **Percentil 95 (P95):** Frontera crítica; superarlo implica niveles de riesgo alto.

        **Condición 1 (Normal):** Todos los gases por debajo del P90 → continuar muestreo normal.

        **Condición 2 (Precaución):** Al menos un gas supera el P90 pero es menor al P95 → aumentar frecuencia de muestreo.

        **Condición 3 (Alta/Alerta):** Algún gas supera el P95 → alta probabilidad de falla activa o reciente; investigación inmediata.
        """)

    st.markdown("---")
    st.markdown("**PASO 1: Clasificación por sistema de preservación** — Ratio O₂/N₂.")
    ieee_o2 = st.number_input("Oxígeno O₂ (µL/L o ppm)", min_value=0.0, value=0.0, key="ieee_o2")
    ieee_n2 = st.number_input("Nitrógeno N₂ (µL/L o ppm)", min_value=0.0, value=10000.0, key="ieee_n2")
    if ieee_n2 and ieee_n2 > 0:
        ratio_val = ieee_o2 / ieee_n2
        ratio_key, sist_label = ieee_paso1_clasificar_sistema(ieee_o2, ieee_n2)
        st.success(f"**Ratio O₂/N₂ = {ratio_val:.4f}** → {sist_label}")
    else:
        ratio_key, sist_label = None, "Ingrese N₂ > 0 para calcular el ratio."
        st.warning(sist_label)

    st.markdown("---")
    st.markdown("**PASO 2: Selección de límites por edad y percentiles**")
    age_sel = st.selectbox(
        "Edad del transformador",
        ["Desconocida", "1–9 años", "10–30 años", ">30 años"],
        key="ieee_age"
    )
    age_map = {"Desconocida": "desc", "1–9 años": "1_9", "10–30 años": "10_30", ">30 años": "30"}
    age_key = age_map[age_sel]

    st.markdown("---")
    st.markdown("**PASO 3: Concentraciones de gas (µL/L o ppm)** — Compare con P90 y P95.")
    g1, g2 = st.columns(2)
    valores_ieee = {}
    with g1:
        for gas in ["H2", "CH4", "C2H6", "C2H4"]:
            valores_ieee[gas] = st.number_input(GASES_LABELS[gas], min_value=0.0, value=0.0, key=f"ieee_{gas}")
    with g2:
        for gas in ["C2H2", "CO", "CO2"]:
            valores_ieee[gas] = st.number_input(GASES_LABELS[gas], min_value=0.0, value=0.0, key=f"ieee_{gas}")

    if ratio_key:
        p90, p95 = ieee_obtener_limites(ratio_key, age_key)
        cond_num, cond_label, cond_rec = ieee_paso3_condicion(valores_ieee, p90, p95)
        if cond_num == 1:
            st.success(f"**{cond_label}**")
        elif cond_num == 2:
            st.warning(f"**{cond_label}**")
        else:
            st.error(f"**{cond_label}**")
        st.info(f"💡 **Recomendación:** {cond_rec}")

        rows = []
        for gas in GASES_IEEE:
            v = valores_ieee.get(gas) or 0
            lim90 = p90.get(gas) or 0
            lim95 = p95.get(gas) or 0
            sup90 = "✓" if v >= lim90 else ""
            sup95 = "✓" if v >= lim95 else ""
            rows.append({"Gas": GASES_LABELS[gas], "Valor (ppm)": v, "P90": lim90, "P95": lim95, "≥ P90": sup90, "≥ P95": sup95})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with st.expander("📋 Ver tablas de referencia (P90 y P95) — IEEE C57.104-2019"):
            st.caption("Tabla 1: Percentil 90 (Condición 1). Tabla 2: Percentil 95 (Condición 2). Valores en µL/L (ppm).")
            for rk, rlabel in [("leq02", "O₂/N₂ ≤ 0.2 (Sellado)"), ("gt02", "O₂/N₂ > 0.2 (Resp. libre)")]:
                st.markdown(f"**{rlabel}**")
                ref = []
                for gas in GASES_IEEE:
                    row = {"Gas": GASES_LABELS[gas]}
                    for ak, alabel in [("desc", "Desc."), ("1_9", "1–9"), ("10_30", "10–30"), ("30", ">30")]:
                        v90 = TABLA_P90[rk][ak].get(gas)
                        v95 = TABLA_P95[rk][ak].get(gas)
                        row[f"P90 ({alabel})"] = v90
                        row[f"P95 ({alabel})"] = v95
                    ref.append(row)
                st.dataframe(pd.DataFrame(ref), width="stretch", hide_index=True)
    else:
        st.caption("Complete O₂ y N₂ (N₂ > 0) para ver límites y condición.")
