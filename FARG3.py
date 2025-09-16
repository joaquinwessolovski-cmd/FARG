import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar lista de equipos
data = pd.read_csv("Partidos 2024-2025.csv")
data = data[['HomeTeam','AwayTeam','HomeGoals','AwayGoals']]
teams = np.unique(np.concatenate([data['HomeTeam'].unique(), data['AwayTeam'].unique()]))
team_idx = {t:i for i,t in enumerate(teams)}

home_ids = data['HomeTeam'].map(team_idx).to_numpy()
away_ids = data['AwayTeam'].map(team_idx).to_numpy()
home_goals = data['HomeGoals'].to_numpy()
away_goals = data['AwayGoals'].to_numpy()

n_teams = len(teams)

# ✅ Cargar parámetros entrenados desde archivo
@st.cache_data
def load_params():
    with open("params.pkl", "rb") as f:
        return pickle.load(f)

params_opt = load_params()

# Separar parámetros
n_teams = len(teams)
attack = params_opt[:n_teams]
defense = params_opt[n_teams:2*n_teams]
home_adv = params_opt[-1]

# Función para calcular intensidades (lambdas)
def expected_goals(home, away):
    i = teams.index(home)
    j = teams.index(away)
    lambda_home = np.exp(home_adv + attack[i] + defense[j])
    lambda_away = np.exp(attack[j] + defense[i])
    return lambda_home, lambda_away

# Función para predecir probabilidades de resultado
def predict_match(home, away, max_goals=6):
    lambda_home, lambda_away = expected_goals(home, away)

    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for hg in range(max_goals+1):
        for ag in range(max_goals+1):
            prob_matrix[hg, ag] = (
                (np.exp(-lambda_home) * lambda_home**hg / np.math.factorial(hg)) *
                (np.exp(-lambda_away) * lambda_away**ag / np.math.factorial(ag))
            )

    home_win = np.sum(np.tril(prob_matrix, -1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.triu(prob_matrix, 1))

    # Resultados más probables
    flat = prob_matrix.flatten()
    top_idx = flat.argsort()[::-1][:5]
    top_scores = [(idx // (max_goals+1), idx % (max_goals+1), flat[idx]) for idx in top_idx]

    return home_win, draw, away_win, prob_matrix, top_scores


# 🚀 Interfaz Streamlit
st.title("Modelo de Poisson (Dixon-Coles) - Predicción de Partidos")

home = st.selectbox("Equipo Local", teams)
away = st.selectbox("Equipo Visitante", teams)

if st.button("Predecir"):
    if home == away:
        st.warning("Elegí equipos distintos.")
    else:
        home_win, draw, away_win, prob_matrix, top_scores = predict_match(home, away)

        st.subheader("Probabilidades del partido:")
        st.write(f"🏠 Victoria {home}: {home_win:.2%}")
        st.write(f"🤝 Empate: {draw:.2%}")
        st.write(f"🚨 Victoria {away}: {away_win:.2%}")

        st.subheader("Marcadores más probables:")
        for hg, ag, p in top_scores:
            st.write(f"{home} {hg} - {ag} {away}: {p:.2%}")

        # Guardar resultados en Excel
        save_file = f"prediccion_{home}_vs_{away}.xlsx"
        with pd.ExcelWriter(save_file, engine="openpyxl") as writer:
            df_probs = pd.DataFrame(prob_matrix, 
                                    index=[f"{i} goles {home}" for i in range(prob_matrix.shape[0])],
                                    columns=[f"{j} goles {away}" for j in range(prob_matrix.shape[1])])
            df_probs.to_excel(writer, sheet_name="Matriz Probabilidades")

            df_top = pd.DataFrame(top_scores, columns=["Goles Local", "Goles Visitante", "Probabilidad"])
            df_top.to_excel(writer, sheet_name="Top Resultados", index=False)

        st.success(f"Resultados guardados en {save_file}")