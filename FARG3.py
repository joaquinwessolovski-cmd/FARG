# Script en Python: Modelo Poisson bivariante + ajuste Dixon-Coles
# Ejecutar en consola o en Visual Studio sin interfaz gráfica.
# Usa datos de 'Partidos 2024-2025.csv' (HomeTeam;AwayTeam;HomeGoals;AwayGoals)

import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import poisson
import math
import streamlit as st
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Cargar datos
data = pd.read_csv("Partidos 2024-2025.csv")

# Columnas necesarias
data = data[['HomeTeam','AwayTeam','HomeGoals','AwayGoals']]
teams = np.unique(np.concatenate([data['HomeTeam'].unique(), data['AwayTeam'].unique()]))
team_idx = {t:i for i,t in enumerate(teams)}

home_ids = data['HomeTeam'].map(team_idx).to_numpy()
away_ids = data['AwayTeam'].map(team_idx).to_numpy()
home_goals = data['HomeGoals'].to_numpy()
away_goals = data['AwayGoals'].to_numpy()

n_teams = len(teams)

# Ajuste Dixon-Coles
def dc_phi(x, y, rho):
    if x == 0 and y == 0:
        return 1 - rho
    if x == 0 and y == 1:
        return 1 + rho
    if x == 1 and y == 0:
        return 1 + rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0

def neg_log_likelihood(params, use_dc=True):
    home_adv = params[0]
    attack = params[1:1+n_teams]
    defense = params[1+n_teams:1+2*n_teams]
    rho = params[-1] if use_dc else 0.0

    pen = 1000.0 * (np.sum(attack))**2 + 1000.0 * (np.sum(defense))**2

    ll = 0.0
    for i in range(len(home_ids)):
        h = home_ids[i]
        a = away_ids[i]
        gh = home_goals[i]
        ga = away_goals[i]
        lambda_h = math.exp(home_adv + attack[h] + defense[a])
        lambda_a = math.exp(attack[a] + defense[h])
        if lambda_h <= 0: 
            lambda_h = 1e-6
        if lambda_a <= 0: 
            lambda_a = 1e-6
        p = poisson.pmf(gh, lambda_h) * poisson.pmf(ga, lambda_a)
        if use_dc:
            p *= dc_phi(gh, ga, rho)
            p = max(p, 1e-15)
        ll += -math.log(p)
    return ll + pen

# Inicialización
init_home_adv = np.log(data['HomeGoals'].mean() + 1e-6) - np.log(data['AwayGoals'].mean() + 1e-6)
init_attack = np.zeros(n_teams)
init_defense = np.zeros(n_teams)
init_rho = 0.0
init_params = np.concatenate(([init_home_adv], init_attack, init_defense, [init_rho]))

# Optimización
bounds = [(0, None)] * len(init_params)
bounds[-1] = (-0.99,0.99)  # rho

res = minimize(neg_log_likelihood, init_params, args=(True,), method='Powell', bounds=bounds)
if not res.success:
    print("Error en la optimización:", res.message)
    exit()

fitted = res.x
home_adv = fitted[0]
attack = fitted[1:1+n_teams]
defense = fitted[1+n_teams:1+2*n_teams]
rho = fitted[-1]

print("Modelo ajustado correctamente.")
print(f"Ventaja de localía (log-scale): {home_adv:.4f}")
print(f"Rho Dixon-Coles: {rho:.4f}")

# --- Predicción de partido ---

def predict_match(home_team, away_team, max_goals=5):
    h_idx = team_idx[home_team]
    a_idx = team_idx[away_team]
    lam_h = math.exp(home_adv + attack[h_idx] + defense[a_idx])
    lam_a = math.exp(attack[a_idx] + defense[h_idx])
    

    probs = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            p *= dc_phi(i,j,rho)
            probs[i,j] = max(p,0)
    probs /= probs.sum()

    home_win = np.sum(np.tril(probs, -1))
    draw = np.sum(np.diag(probs))
    away_win = np.sum(np.triu(probs, 1))
    
    scores = [(i, j, probs[i, j]) for i in range(probs.shape[0]) for j in range(probs.shape[1])]
    scores_sorted = sorted(scores, key=lambda x: -x[2])[:5]
    
    return (home_win, draw, away_win, probs, scores_sorted)





######








st.title("Predicción de Resultados - Poisson Dixon-Coles")

col1, col2 = st.columns([1,2])

with col1:
    home = st.selectbox("Equipo Local", teams)
    away = st.selectbox("Equipo Visitante", teams)
    if st.button("Predecir"):
        if home == away:
            st.warning("El equipo local y visitante deben ser distintos")
        else:
            home_win, draw, away_win, probs, top_scores = predict_match(home, away)
            
            # Mostrar resultados
            st.subheader(f"{home} vs {away}")
            st.write(f"Probabilidad {home}: {home_win:.2%}")
            st.write(f"Probabilidad Empate: {draw:.2%}")
            st.write(f"Probabilidad {away}: {away_win:.2%}")

            st.write("Top 5 marcadores más probables:")
            for s in top_scores:
                st.write(f"{s[0]} - {s[1]} : {s[2]:.2%}")

            # Guardar resultados en Excel
            save_file = "Predicciones.xlsx"
            df_out = pd.DataFrame({
                "Local": [home],
                "Visitante": [away],
                "Prob_Local": [home_win],
                "Prob_Empate": [draw],
                "Prob_Visitante": [away_win],
                "Top5_Marcadores": [", ".join([f'{s[0]}-{s[1]} ({s[2]:.2%})' for s in top_scores])]
            })

            if os.path.exists(save_file):
                with pd.ExcelWriter(save_file, mode="a", if_sheet_exists="overlay") as writer:
                    if "Predicciones" not in writer.sheets:
                        df_out.to_excel(writer, sheet_name="Predicciones", index=False)
                    else:
                        startrow = writer.sheets["Predicciones"].max_row
                        df_out.to_excel(writer, sheet_name="Predicciones", index=False, header=False, startrow=startrow)
            else:
                with pd.ExcelWriter(save_file) as writer:
                    df_out.to_excel(writer, sheet_name="Predicciones", index=False)

            # Mostrar heatmap
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(probs, cmap="Blues")
            ax.set_title("Distribución de Goles")
            ax.set_xlabel("Goles Visitante")
            ax.set_ylabel("Goles Local")
            fig.colorbar(im, ax=ax)
            col2.pyplot(fig)
