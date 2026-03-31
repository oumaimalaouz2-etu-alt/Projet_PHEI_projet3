import numpy as np
from scipy.optimize import curve_fit

# Paramètres
N0 = 1000
lambda_real = 0.05
t_max = 100
nb_points = 50
bruit_sigma = 20

# Génération données
t = np.linspace(0, t_max, nb_points)
N_theo = N0 * np.exp(-lambda_real * t)
np.random.seed(42)
N_data = N_theo + np.random.normal(0, bruit_sigma, nb_points)
N_data = np.maximum(N_data, 0)

# Fonction modèle
def modele(t, N0_est, lambda_est):
    return N0_est * np.exp(-lambda_est * t)

popt, pcov = curve_fit(modele, t, N_data, p0=[1000, 0.05])

lambda_est = popt[1]
erreur_lambda = np.sqrt(np.diag(pcov))[1]

print(f"λ estimé = {lambda_est:.4f} ± {erreur_lambda:.4f}")
print(f"Demi-vie estimée = {np.log(2)/lambda_est:.2f} unités de temps")
print(f"λ réel = {lambda_real:.4f}")
print(f"Erreur relative sur λ = {abs(lambda_est - lambda_real)/lambda_real * 100:.2f}%")