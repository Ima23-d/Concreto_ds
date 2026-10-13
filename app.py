import io
import os
import urllib.request

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Configuração da página
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Previsão da Resistência do Concreto", page_icon="🏗️", layout="wide")

COLUNAS = ["cimento", "escoria", "cinza", "agua", "superplast",
           "agregado_graudo", "agregado_miudo", "idade", "resistencia"]
URL_DATASET = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
URL_IMAGEM = "https://i.postimg.cc/nzYgGKgB/images.jpg"


# ----------------------------------------------------------------------------
# Funções de dados e modelo (com cache para o app ficar rápido)
# ----------------------------------------------------------------------------
@st.cache_data
def carregar_dados(arquivo_bytes=None):
    """Lê o Excel enviado, o arquivo local ou baixa do UCI."""
    if arquivo_bytes is not None:
        return pd.read_excel(io.BytesIO(arquivo_bytes))
    if not os.path.exists("Concrete_Data.xls"):
        urllib.request.urlretrieve(URL_DATASET, "Concrete_Data.xls")
    return pd.read_excel("Concrete_Data.xls")


@st.cache_data
def preparar_dados(df_bruto):
    """Renomeia colunas, remove duplicados e cria as novas variáveis."""
    df = df_bruto.copy()
    df.columns = COLUNAS
    n_duplicados = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df["log_idade"] = np.log(df["idade"])
    df["agua_cimento"] = df["agua"] / df["cimento"]
    return df, n_duplicados


@st.cache_data
def treinar_modelo(df):
    """Executa toda a modelagem e devolve os resultados."""
    X = df.drop(columns=["resistencia", "idade"])
    y = df["resistencia"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

    # Validação cruzada (5 folds) com alpha = 1.0
    scores_cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

    # Busca do melhor alpha (somente dados de treino)
    parametros = {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}
    busca = GridSearchCV(pipe, parametros, cv=5, scoring="r2")
    busca.fit(X_train, y_train)
    pipe = busca.best_estimator_

    # Avaliação no teste
    y_pred = pipe.predict(X_test)

    # Diagnóstico de overfitting e underfitting
    modelos = {
        "Baseline (média)": DummyRegressor(),
        "Regressão Linear": LinearRegression(),
        "Ridge (final)": Ridge(alpha=pipe.named_steps["ridge"].alpha),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    linhas = []
    for nome, modelo in modelos.items():
        m = Pipeline([("scaler", StandardScaler()), ("modelo", modelo)])
        m.fit(X_train, y_train)
        r2_tr = r2_score(y_train, m.predict(X_train))
        r2_te = r2_score(y_test, m.predict(X_test))
        rmse_te = mean_squared_error(y_test, m.predict(X_test)) ** 0.5
        linhas.append([nome, r2_tr, r2_te, r2_tr - r2_te, rmse_te])
    comparacao = pd.DataFrame(
        linhas, columns=["Modelo", "R² treino", "R² teste", "Diferença", "RMSE teste (MPa)"])

    # Deploy: serializa o pipeline treinado em memória (evita erro de arquivo no Windows)
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    modelo_bytes = buffer.getvalue()

    return {
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "pipe": pipe, "scores_cv": scores_cv,
        "alphas": [str(a) for a in parametros["ridge__alpha"]],
        "cv_alphas": busca.cv_results_["mean_test_score"],
        "melhor_alpha": busca.best_params_["ridge__alpha"],
        "melhor_score": busca.best_score_,
        "y_pred": y_pred,
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_test, y_pred),
        "comparacao": comparacao,
        "modelo_bytes": modelo_bytes,
    }


# ----------------------------------------------------------------------------
# Carregamento dos dados
# ----------------------------------------------------------------------------
st.sidebar.title("🏗️ Concreto")
st.sidebar.markdown("Projeto completo de Machine Learning, da preparação dos dados ao deploy.")
arquivo = st.sidebar.file_uploader("Opcional: envie o Concrete_Data.xls", type=["xls", "xlsx"])

try:
    df_bruto = carregar_dados(arquivo.getvalue() if arquivo else None)
except Exception as e:
    st.error(f"Não foi possível carregar o dataset: {e}")
    st.info("Envie o arquivo Concrete_Data.xls pela barra lateral ou coloque-o na mesma pasta do app.py.")
    st.stop()

df, n_duplicados = preparar_dados(df_bruto)
res = treinar_modelo(df)
pipe = res["pipe"]

# Dados brutos com as colunas renomeadas (como no notebook, antes de remover duplicados)
df_renomeado = df_bruto.copy()
df_renomeado.columns = COLUNAS

st.sidebar.success(f"Dataset: {len(df_bruto)} linhas → {len(df)} após limpeza")
st.sidebar.metric("R² no teste", f"{res['r2']:.2f}")
st.sidebar.metric("RMSE no teste", f"{res['rmse']:.2f} MPa")

st.title("🏗️ Previsão da Resistência do Concreto")

abas = st.tabs([
    "1. Visão geral",
    "2. Dados e limpeza",
    "3. Outliers",
    "4. Atributos e correlação",
    "5. Modelagem",
    "6. Avaliação",
    "7. Deploy e simulador",
    "8. Conclusão",
])

# ----------------------------------------------------------------------------
# 1. Visão geral
# ----------------------------------------------------------------------------
with abas[0]:
   
    st.header("Entendimento do Negócio")
    st.write(
        "O processo envolve a fabricação e controle de qualidade do concreto na engenharia civil, "
        "onde é necessário determinar a resistência à compressão (em MPa) de diferentes misturas de "
        "dosagem e idades de cura para garantir a segurança estrutural.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Problema")
        st.write(
            "A forma tradicional exige esperar 28 dias de cura para que uma prensa meça a resistência, "
            "o que gera riscos de prejuízos milionários caso a estrutura falhe, além do desperdício de "
            "cimento e tempo de laboratório em ajustes por tentativa e erro.")
    with c2:
        st.subheader("Solução")
        st.write(
            "Um modelo de aprendizado de máquina (Regressão Ridge) integrado a um aplicativo em Streamlit, "
            "que recebe os ingredientes da dosagem e a idade de cura para fornecer uma estimativa "
            "instantânea da resistência em MPa antes da fabricação física.")

    st.subheader("Etapas do projeto")
    st.markdown(
        "- **Dados e limpeza:** tipos, nulos, duplicados e valores zero\n"
        "- **Outliers:** contagem por IQR, boxplots e decisão\n"
        "- **Atributos e correlação:** `log_idade`, `agua_cimento`, Pearson, matriz e relações principais\n"
        "- **Modelagem:** split, Pipeline (StandardScaler + Ridge), validação cruzada e ajuste do alpha\n"
        "- **Avaliação:** R², RMSE, MAE, resíduos e overfitting/underfitting\n"
        "- **Deploy:** modelo salvo com joblib e simulador de previsão\n"
        "- **Conclusão:** impacto, limitações e próximos passos")
    st.caption("Dataset: Concrete_Data.xls (UCI Concrete Compressive Strength)")

# ----------------------------------------------------------------------------
# 2. Dados e limpeza
# ----------------------------------------------------------------------------
with abas[1]:
    st.header("Preparação e Limpeza dos Dados")

    st.subheader("Leitura e renomeação das colunas")
    st.write(f"O arquivo bruto tem **{df_bruto.shape[0]} linhas** e **{df_bruto.shape[1]} colunas**, "
             "renomeadas para o português.")
    st.dataframe(df_renomeado.head(), use_container_width=True)

    st.subheader("Tipos de variáveis")
    tipos = df_renomeado.dtypes.astype(str).rename("tipo").to_frame()
    tipos["não nulos"] = df_renomeado.notnull().sum()
    st.dataframe(tipos, use_container_width=True)
    st.caption("Todas as colunas são numéricas (8 float e 1 int), então não é necessário codificar variáveis.")

    st.subheader("Valores nulos")
    nulos = df_renomeado.isnull().sum()
    c1, c2 = st.columns([1, 2])
    c1.metric("Total de nulos", int(nulos.sum()))
    c2.dataframe(nulos.rename("nulos").to_frame().T, use_container_width=True)

    st.subheader("Remoção de duplicados")
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas antes", len(df_bruto))
    c2.metric("Duplicadas removidas", n_duplicados)
    c3.metric("Linhas depois", len(df))

    st.subheader("Valores zero")
    zeros = (df[COLUNAS] == 0).sum().rename("zeros").to_frame().T
    st.dataframe(zeros, use_container_width=True)
    st.caption("Zero em escória, cinza e superplastificante significa ingrediente ausente na receita (não é erro).")

    st.subheader("Estatísticas descritivas")
    st.dataframe(df[COLUNAS].describe(), use_container_width=True)

# ----------------------------------------------------------------------------
# 3. Outliers
# ----------------------------------------------------------------------------
with abas[2]:
    st.header("Análise de Outliers")
    st.write("Contagem de outliers pelo critério IQR (1.5 × IQR) em todas as colunas, "
             "seguida de um boxplot por variável (as escalas são diferentes).")

    dados = df[COLUNAS]
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1
    outliers = ((dados < q1 - 1.5 * iqr) | (dados > q3 + 1.5 * iqr)).sum()

    c1, c2 = st.columns([1, 2])
    c1.dataframe(outliers.rename("outliers"), use_container_width=True)

    # Boxplots (um por coluna, grade 3 x 3)
    st.subheader("Boxplot de cada variável")
    for i in range(0, len(COLUNAS), 3):
        cols = st.columns(3)
        for col_st, nome in zip(cols, COLUNAS[i:i + 3]):
            box = alt.Chart(dados[[nome]]).mark_boxplot(color="teal", extent=1.5).encode(
                y=alt.Y(f"{nome}:Q", title=nome)
            ).properties(title=nome, height=250)
            col_st.altair_chart(box, use_container_width=True)

    st.subheader("Análise de Outliers: Diagnóstico e Tratamento")
    lista = "\n".join(
        f"* **{nome}**: {int(qtd)} outliers"
        for nome, qtd in outliers.sort_values(ascending=False).items() if qtd > 0
    )
    st.markdown(lista)

    st.markdown("#### O tratamento atual está correto?")
    st.markdown(
        "**Sim, a decisão de não removê-los é tecnicamente correta para este problema pelas seguintes razões:**\n\n"
        "1. **Natureza dos Dados (Engenharia Civil):** Valores altos de idade (como cura de 1 ano / 365 dias) "
        "ou dosagens extremas de superplastificante e água são perfeitamente possíveis e representam receitas "
        "reais de concreto especiais. Não se tratam de erros de digitação ou falhas de medição.\n"
        "2. **Uso de Regressão Ridge:** A escolha do modelo **Ridge (L2 Regularization)** é excelente para este "
        "cenário. A penalização Ridge ajuda a suavizar o impacto de variáveis com alta variância e reduz a "
        "sensibilidade do modelo a observações extremas (outliers), mitigando o risco de overfitting sem a "
        "necessidade de descartar dados valiosos.")

# ----------------------------------------------------------------------------
# 4. Atributos e correlação
# ----------------------------------------------------------------------------
with abas[3]:
    st.header("Engenharia de Atributos e Análise de Correlação")

    st.subheader("Criação de novas variáveis")
    c1, c2 = st.columns(2)
    c1.markdown("**`log_idade`**: aplicação do logaritmo à idade para linearizar o ganho de força ao longo do tempo.")
    c1.code('df["log_idade"] = np.log(df["idade"])', language="python")
    c2.markdown("**`agua_cimento`**: cálculo da razão água/cimento com base na Lei de Abrams.")
    c2.code('df["agua_cimento"] = df["agua"] / df["cimento"]', language="python")
    st.dataframe(df[["idade", "log_idade", "agua", "cimento", "agua_cimento"]].head(), use_container_width=True)

    st.markdown(
        "* **`log_idade` (Logaritmo da Idade):** O concreto ganha muita resistência nos primeiros dias e quase "
        "estabiliza depois. Modelos lineares não entendem esse comportamento curvado. Aplicar o logaritmo "
        "transforma essa curva exponencial em uma linha reta, facilitando o aprendizado do modelo.\n\n"
        "* **`agua_cimento` (Razão Água/Cimento):** Pela física da construção (Lei de Abrams), esta proporção "
        "é o fator que mais dita a força do concreto (mais água enfraquece, mais cimento fortalece). Como "
        "modelos lineares não conseguem dividir uma coluna pela outra sozinhos, nós já entregamos esse cálculo "
        "pronto para ele.")

    st.subheader("Análise de correlação")
    st.write("Estudo da Correlação de Pearson para identificar quais as variáveis que mais se correlacionam "
             "positivamente (ex: `log_idade`, `cimento`) ou negativamente (ex: `agua_cimento`, `agua`) "
             "com a resistência.")

    corr = df.corr()["resistencia"].drop("resistencia").sort_values()
    df_corr = corr.reset_index()
    df_corr.columns = ["Variável", "Correlação"]

    chart_corr = alt.Chart(df_corr).mark_bar().encode(
        x=alt.X("Correlação:Q", title="Coeficiente de Correlação"),
        y=alt.Y("Variável:N", sort="-x", title="Variáveis"),
        color=alt.condition(
            alt.datum.Correlação > 0,
            alt.value("teal"),
            alt.value("chocolate")
        )
    ).properties(title="Correlação de Pearson com a Resistência (MPa)", height=300)
    st.altair_chart(chart_corr, use_container_width=True)

    st.subheader("Matriz de correlação entre as variáveis")
    matriz_corr = df.corr().reset_index().melt(id_vars="index")
    matriz_corr.columns = ["Variável 1", "Variável 2", "Correlação"]

    heatmap = alt.Chart(matriz_corr).mark_rect().encode(
        x=alt.X("Variável 2:N", title=""),
        y=alt.Y("Variável 1:N", title=""),
        color=alt.Color("Correlação:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]))
    ).properties(height=400)

    text = heatmap.mark_text(baseline="middle").encode(
        text=alt.Text("Correlação:Q", format=".2f"),
        color=alt.condition(
            (alt.datum.Correlação > 0.5) | (alt.datum.Correlação < -0.5),
            alt.value("white"),
            alt.value("black")
        )
    )
    st.altair_chart(heatmap + text, use_container_width=True)
    st.markdown("Relações principais: **Cimento, razão água/cimento e log da idade contra a resistência.** "
                "O heatmap mostra também a correlação entre as variáveis de entrada: `agua`, `cimento` e "
                "`agua_cimento` são relacionadas entre si (multicolinearidade), o que justifica a "
                "penalização L2 do Ridge.")

    st.subheader("Relações principais com a resistência")
    c1, c2, c3 = st.columns(3)
    relacoes = [
        (c1, "cimento", "Cimento (kg/m³)", "Cimento vs Resistência", "blue"),
        (c2, "agua_cimento", "Razão Água/Cimento", "Água/Cimento vs Resistência", "chocolate"),
        (c3, "log_idade", "Log(Idade em dias)", "Log da Idade vs Resistência", "green"),
    ]
    for coluna_st, var, rotulo, titulo, cor in relacoes:
        graf = alt.Chart(df[[var, "resistencia"]]).mark_circle(opacity=0.6, color=cor).encode(
            x=alt.X(f"{var}:Q", title=rotulo, scale=alt.Scale(zero=False)),
            y=alt.Y("resistencia:Q", title="Resistência (MPa)")
        ).properties(title=f"{titulo} (r = {corr[var]:+.2f})", height=300)
        coluna_st.altair_chart(graf, use_container_width=True)

# ----------------------------------------------------------------------------
# 5. Modelagem
# ----------------------------------------------------------------------------
with abas[4]:
    st.header("Modelagem (Preparação e Treino)")

    st.subheader("Separação de variáveis")
    st.write("Definição das variáveis preditoras de entrada **X** (excluindo a resistência e a idade bruta) "
             "e da variável alvo **y** (resistência em MPa).")
    st.code('X = df.drop(columns=["resistencia", "idade"])\ny = df["resistencia"]', language="python")
    st.write("Colunas de X:", ", ".join(res["X"].columns))

    st.subheader("Divisão treino e teste (80% / 20%)")
    st.code('X_train, X_test, y_train, y_test = train_test_split(\n'
            '    X, y, test_size=0.2, random_state=42\n)', language="python")
    c1, c2 = st.columns(2)
    c1.metric("Treino", f"{len(res['X_train'])} linhas")
    c2.metric("Teste", f"{len(res['X_test'])} linhas")

    st.subheader("Pipeline: StandardScaler + Ridge")
    st.write("Fluxo que inclui a normalização de escala com StandardScaler (média 0 e desvio padrão 1) "
             "e o modelo de Regressão Ridge com penalização L2 (`Ridge(alpha=1.0)`) para controlar o overfitting. "
             "O scaler fica dentro do Pipeline, então a validação cruzada não tem vazamento de dados.")
    st.code('pipe = Pipeline([\n    ("scaler", StandardScaler()),\n    ("ridge", Ridge(alpha=1.0))\n])',
            language="python")

    st.subheader("Validação cruzada (5 folds)")
    scores = res["scores_cv"]
    c1, c2 = st.columns([1, 2])
    c1.metric("R² médio (CV)", f"{scores.mean():.2f}", f"± {scores.std() * 2:.2f}", delta_color="off")
    c1.caption("Scores por fold: " + ", ".join(f"{s:.2f}" for s in scores))

    df_folds = pd.DataFrame({
        "Fold": [f"Fold {i + 1}" for i in range(len(scores))],
        "R²": scores
    })
    chart_folds = alt.Chart(df_folds).mark_bar(color="steelblue").encode(
        x="Fold:N",
        y=alt.Y("R²:Q", scale=alt.Scale(domain=[0, 1]))
    ).properties(height=200)
    c2.altair_chart(chart_folds, use_container_width=True)

    st.subheader("Ajuste do hiperparâmetro alpha")
    st.write("Busca do melhor `alpha` do Ridge por validação cruzada (5 folds), usando apenas os dados de treino. "
             "O modelo final é o pipeline com o melhor alpha.")
    df_alpha = pd.DataFrame({
        "alpha": res["alphas"],
        "R2_medio": res["cv_alphas"]
    })
    chart_alpha = alt.Chart(df_alpha).mark_line(point=True).encode(
        x=alt.X("alpha:N", title="alpha", sort=None),
        y=alt.Y("R2_medio:Q", title="R² médio (CV)", scale=alt.Scale(zero=False))
    ).properties(title="Busca do melhor alpha (GridSearchCV, somente treino)", height=250)
    st.altair_chart(chart_alpha, use_container_width=True)
    st.success(f"Melhor alpha: **{res['melhor_alpha']}** | R² na validação cruzada: **{res['melhor_score']:.2f}**. "
               "O modelo final é o pipeline com esse alpha, treinado nos dados de treino.")

# ----------------------------------------------------------------------------
# 6. Avaliação
# ----------------------------------------------------------------------------
with abas[5]:
    st.header("Avaliação do Modelo")
    y_test, y_pred = res["y_test"], res["y_pred"]
    residuos = y_test - y_pred

    st.subheader("Métricas no conjunto de teste")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{res['r2']:.2f}", help="Variância da resistência explicada pelo modelo")
    c2.metric("RMSE", f"{res['rmse']:.2f} MPa", help="Erro típico, na mesma unidade do alvo")
    c3.metric("MAE", f"{res['mae']:.2f} MPa", help="Erro absoluto médio, sem peso extra para erros grandes")
    c4.metric("Resíduo médio", f"{residuos.mean():+.2f} MPa", help="Deve estar perto de zero")

    st.subheader("Real vs previsto e resíduos")
    c1, c2 = st.columns(2)

    df_scatter = pd.DataFrame({"Real": y_test, "Previsto": y_pred})
    chart_scatter = alt.Chart(df_scatter).mark_circle(opacity=0.6, color="steelblue").encode(
        x=alt.X("Real:Q", title="Resistência real (MPa)"),
        y=alt.Y("Previsto:Q", title="Resistência prevista (MPa)")
    ).properties(title=f"Real vs Previsto (R² = {res['r2']:.2f})", height=300)

    # Mesma linha de referência do notebook: de y.min() a y.max()
    min_val, max_val = res["y"].min(), res["y"].max()
    df_linha = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})
    line = alt.Chart(df_linha).mark_line(color="black").encode(x="x:Q", y="y:Q")
    c1.altair_chart(chart_scatter + line, use_container_width=True)

    df_hist = pd.DataFrame({"Resíduo": residuos})
    chart_hist = alt.Chart(df_hist).mark_bar(color="steelblue").encode(
        x=alt.X("Resíduo:Q", bin=alt.Bin(maxbins=25), title="Resíduo: real - previsto (MPa)"),
        y=alt.Y("count()", title="Amostras")
    ).properties(title=f"Resíduos (RMSE = {res['rmse']:.1f} MPa)", height=300)
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
    c2.altair_chart(chart_hist + zero_line, use_container_width=True)

    st.subheader("Diagnóstico de overfitting e underfitting")
    st.write("Comparação do R² de treino e de teste do Ridge com um baseline (média), a Regressão Linear e um "
             "modelo não linear (Gradient Boosting). Treino muito acima do teste indica overfitting; "
             "treino e teste baixos indicam underfitting.")
    comp = res["comparacao"]
    st.dataframe(comp.round(2), use_container_width=True)

    comp_melted = comp.melt(id_vars="Modelo", value_vars=["R² treino", "R² teste"], var_name="Dataset", value_name="R²")
    chart_comp = alt.Chart(comp_melted).mark_bar().encode(
        x=alt.X("Modelo:N", title=""),
        y=alt.Y("R²:Q", title="R²"),
        color=alt.Color("Dataset:N", scale=alt.Scale(range=["steelblue", "chocolate"])),
        column="Dataset:N"
    ).properties(height=250)
    st.altair_chart(chart_comp, use_container_width=True)

    st.info("O Ridge tem R² de treino e de teste praticamente iguais, então **não há overfitting**. "
            "Porém o ganho do Gradient Boosting mostra que a relação tem partes não lineares que o "
            "modelo linear não captura (**underfitting leve**). O Ridge segue como modelo final por ser "
            "simples, interpretável e fácil de levar ao deploy; modelos não lineares ficam como próximo passo.")

    st.subheader("Coeficientes do Ridge (variáveis padronizadas)")
    coef = pd.Series(pipe.named_steps["ridge"].coef_, index=res["X"].columns).sort_values()
    df_coef = coef.reset_index()
    df_coef.columns = ["Variável", "Coeficiente"]

    chart_coef = alt.Chart(df_coef).mark_bar().encode(
        x=alt.X("Coeficiente:Q", title="Coeficiente"),
        y=alt.Y("Variável:N", sort="-x", title=""),
        color=alt.condition(
            alt.datum.Coeficiente > 0,
            alt.value("steelblue"),
            alt.value("chocolate")
        )
    ).properties(height=300)
    st.altair_chart(chart_coef, use_container_width=True)

# ----------------------------------------------------------------------------
# 7. Deploy e simulador
# ----------------------------------------------------------------------------
with abas[6]:
    st.header("Deploy")
    st.write("Exportação do pipeline completo e treinado (scaler + Ridge) para um ficheiro externo "
             "(`modelo_concreto.joblib`) utilizando a biblioteca joblib. Ele é usado pelo simulador abaixo.")
    st.code("joblib.dump(pipe, 'modelo_concreto.joblib')", language="python")
    st.download_button("⬇️ Baixar modelo_concreto.joblib", res["modelo_bytes"],
                       file_name="modelo_concreto.joblib")

    st.divider()
    st.header("Simulador de Resistência")
    st.write("Informe a dosagem e a idade de cura para estimar a resistência em MPa.")

    c1, c2, c3 = st.columns(3)
    with c1:
        cimento = st.slider("Cimento (kg/m³)", 100.0, 550.0, 280.0, 10.0)
        escoria = st.slider("Escória de Alto Forno (kg/m³)", 0.0, 360.0, 0.0, 10.0)
        cinza = st.slider("Cinza Volante (kg/m³)", 0.0, 200.0, 0.0, 10.0)
    with c2:
        agua = st.slider("Água (kg/m³)", 120.0, 250.0, 180.0, 5.0)
        superplast = st.slider("Superplastificante (kg/m³)", 0.0, 35.0, 0.0, 1.0)
        idade = st.slider("Idade de Cura (dias)", 1, 365, 28)
    with c3:
        agregado_graudo = st.slider("Agregado Graúdo/Brita (kg/m³)", 800.0, 1150.0, 970.0, 10.0)
        agregado_miudo = st.slider("Agregado Miúdo/Areia (kg/m³)", 590.0, 1000.0, 770.0, 10.0)

    def montar_entrada(dias):
        """Recria as mesmas features do treino (mesma ordem de colunas)."""
        entrada = pd.DataFrame([{
            "cimento": cimento, "escoria": escoria, "cinza": cinza, "agua": agua,
            "superplast": superplast, "agregado_graudo": agregado_graudo,
            "agregado_miudo": agregado_miudo,
            "log_idade": np.log(dias), "agua_cimento": agua / cimento,
        }])
        return entrada[list(res["X"].columns)]

    st.metric("Razão água/cimento", f"{agua / cimento:.2f}")

    if st.button("💡 Calcular Resistência Estimada"):
        predicao = max(0.0, pipe.predict(montar_entrada(idade))[0])
        if predicao < 20:
            st.error(f"### Resistência Estimada: **{predicao:.2f} MPa**\n\n"
                     "⚠️ Abaixo de 20 MPa, mínimo usual para concreto estrutural.")
        elif predicao < 30:
            st.warning(f"### Resistência Estimada: **{predicao:.2f} MPa**\n\n")
        else:
            st.success(f"### Resistência Estimada: **{predicao:.2f} MPa**")

        dias_lista = [1, 3, 7, 14, 28, 56, 90, 180, 365]
        valores_evolucao = [max(0.0, pipe.predict(montar_entrada(d))[0]) for d in dias_lista]

        df_evolucao = pd.DataFrame({
            "Idade": dias_lista,
            "Resistencia": valores_evolucao
        })

        chart_evolucao = alt.Chart(df_evolucao).mark_line(point=True, color="teal").encode(
            x=alt.X("Idade:O", title="Idade (dias)"),
            y=alt.Y("Resistencia:Q", title="Resistência estimada (MPa)")
        ).properties(title="Evolução da resistência da mesma mistura", height=300)

        st.altair_chart(chart_evolucao, use_container_width=True)
        st.caption("Estimativa baseada nos dados experimentais históricos de laboratório do dataset UCI Concrete.")

# ----------------------------------------------------------------------------
# 8. Conclusão
# ----------------------------------------------------------------------------
with abas[7]:
    st.header("Conclusão")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Impacto")
        st.markdown(
            "- Decisão mais rápida, sem esperar os 28 dias de cura.\n"
            "- Menos desperdício de cimento e de tempo de laboratório.\n"
            "- Conformidade estrutural verificada antes da obra.")
    with c2:
        st.subheader("Limitações e próximos passos")
        st.markdown(
            "- O dataset não tem temperatura nem umidade de cura.\n"
            "- O modelo linear (Ridge) subestima relações não lineares; testar modelos como Gradient Boosting ou Random Forest.\n"
            "- O app deve ser usado dentro das faixas de dosagem do dataset original (previsões fora dessas faixas não são confiáveis).")