import numpy as np
import matplotlib.pyplot as plt

# ======================================
# 1) Paramètres réels (simulation)
# ======================================
N0 = 1000          # nombre initial
lambda_true = 0.3  # constante de décroissance (s^-1)

# Axe du temps
t = np.linspace(0, 10, 50)

# ======================================
# 2) Génération des données exponentielles
# ======================================
N_theorique = N0 * np.exp(-lambda_true * t)

# ======================================
# 3) Ajout d’un bruit expérimental
# ======================================
np.random.seed(0)
sigma = 40
bruit = np.random.normal(0, sigma, size=t.size)
N_mes = N_theorique + bruit

# Eviter valeurs négatives
N_mes = np.clip(N_mes, 1e-6, None)

# ======================================
# 4) Visualisation
# ======================================
plt.figure()
plt.scatter(t, N_mes, label="Données bruitées")
plt.plot(t, N_theorique, label="Modèle théorique")
plt.xlabel("Temps (s)")
plt.ylabel("N(t)")
plt.title("Décroissance radioactive simulée")
plt.legend()
plt.grid()
plt.show()

# ======================================
# 5) Ajustement simple (linéarisation)
# ln(N) = ln(N0) - lambda*t
# ======================================
y = np.log(N_mes)

a, b = np.polyfit(t, y, 1)  # y = a*t + b

lambda_est = -a
N0_est = np.exp(b)

# Demi-vie estimée
T12_est = np.log(2) / lambda_est

print("===== Résultats =====")
print("lambda vrai      =", lambda_true)
print("lambda estimé    =", lambda_est)
print("N0 estimé        =", N0_est)
print("Demi-vie estimée =", T12_est)
