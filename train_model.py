import pickle
from scipy.optimize import minimize 
from scipy.stats import poisson
import pandas as pd
import numpy as np
import math

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

res = minimize(neg_log_likelihood, init_params, args=(True), method='Powell', bounds=bounds)
if not res.success:
    print("Error en la optimización:", res.message)
    exit()

fitted = res.x
home_adv = fitted[0]
attack = fitted[1:1+n_teams]
defense = fitted[1+n_teams:1+2*n_teams]
rho = fitted[-1]

# Guardar a archivo
with open("params.pkl", "wb") as f:
    pickle.dump(fitted, f)

print("✅ Modelo entrenado y parámetros guardados en params.pkl")
