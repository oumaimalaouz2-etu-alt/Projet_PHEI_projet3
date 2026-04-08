import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from random import randint


class Physique:
    """Modèle physique : N(t) = N0 * exp(-λ*t) avec bruit"""
    
    def __init__(self, N0, lambd, temps_max, points):
        self.N0 = N0
        self.lambd = lambd
        self.temps_max = temps_max
        self.points = points
    
    def loi_exp(self, t, N0, lambd):
        return N0 * np.exp(-lambd * t)
    
    def simuler(self):
        # Génère les données
        self.t = np.linspace(0, self.temps_max, self.points)
        self.theorie = self.loi_exp(self.t, self.N0, self.lambd)
        
        # Ajoute du bruit
        bruit = np.random.normal(0, 0.05, self.points)
        self.mesures = self.theorie * (1 + bruit)
        self.mesures = np.maximum(self.mesures, 1)  # Pas de négatif
        
        # Ajuste le modèle
        params, _ = curve_fit(self.loi_exp, self.t, self.mesures, p0=[900, 0.1])
        self.N0_trouve, self.lambd_trouve = params
        self.T12 = np.log(2) / self.lambd_trouve
        
        return self.t, self.mesures
    
    def lineariser(self):
        """Transforme ln(N) pour obtenir une droite"""
        lnN = np.log(self.mesures)
        pente, ordonnee = np.polyfit(self.t, lnN, 1)
        return lnN, -pente  # -pente = λ


class Stochastique:
    """Simulation avec des dés (1 chance sur 6 de mourir)"""
    
    def __init__(self, N0):
        self.N0 = N0
    
    def simuler(self):
        noyaux = [1] * self.N0  # 1 = vivant
        self.N_hist = [self.N0]
        self.t_hist = [0]
        
        while noyaux:
            # Chaque noyau lance un dé : il meurt si = 6
            noyaux = [1 for _ in noyaux if randint(1, 6) != 6]
            self.N_hist.append(len(noyaux))
            self.t_hist.append(self.t_hist[-1] + 1)
        
        # Enlève le dernier point (N=0)
        self.t = np.array(self.t_hist[:-1])
        self.N = np.array(self.N_hist[:-1])
        
        # Ajuste exponentielle
        def expo(t, N0, lam):
            return N0 * np.exp(-lam * t)
        
        params, _ = curve_fit(expo, self.t, self.N, p0=[self.N0, 0.1])
        self.N0_trouve, self.lambd_trouve = params
        self.T12 = np.log(2) / self.lambd_trouve
        
        # Théorique
        self.T12_theorique = np.log(2) / (-np.log(5/6))
        
        return self.t, self.N



class Graphique:
    """Dessine les résultats"""
    
    @staticmethod
    def dessiner(physique, stochastique):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Graphique 1 : Physique
        ax1.scatter(physique.t, physique.mesures, alpha=0.7, label='Mesures')
        ax1.plot(physique.t, physique.theorie, 'r--', label='Théorie')
        ax1.plot(physique.t, physique.loi_exp(physique.t, physique.N0_trouve, physique.lambd_trouve), 
                'g-', label=f'Ajustement λ={physique.lambd_trouve:.3f}')
        ax1.axvline(physique.T12, color='orange', linestyle=':', label=f'T½={physique.T12:.1f}s')
        ax1.set_xlabel('Temps (s)')
        ax1.set_ylabel('N(t)')
        ax1.set_title('Modèle Physique')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2 : Stochastique
        ax2.step(stochastique.t, stochastique.N, where='post', alpha=0.7, label='Simulation')
        ax2.plot(stochastique.t, physique.loi_exp(stochastique.t, stochastique.N0_trouve, stochastique.lambd_trouve),
                'g-', label=f'λ={stochastique.lambd_trouve:.3f}')
        ax2.axvline(stochastique.T12, color='orange', linestyle=':', label=f'T½={stochastique.T12:.1f}')
        ax2.set_xlabel('Temps (pas)')
        ax2.set_ylabel('N(t)')
        ax2.set_title('Modèle Stochastique (dés)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()



# 1. Crée et simule le modèle physique
print(" Simulation physique...")
phys = Physique(N0=1000, lambd=0.10, temps_max=60, points=40)
phys.simuler()

# 2. Crée et simule le modèle stochastique
print(" Simulation avec dés...")
stoch = Stochastique(N0=1000)
stoch.simuler()

# 3. Résultats
print("\n" + "="*50)
print("RÉSULTATS")
print("="*50)
print(f"\n PHYSIQUE :")
print(f"   λ réel    = 0.1000")
print(f"   λ trouvé  = {phys.lambd_trouve:.4f}")
print(f"   Demi-vie  = {phys.T12:.2f} s")

print(f"\n STOCHASTIQUE :")
print(f"   T½ théorique = {stoch.T12_theorique:.2f}")
print(f"   T½ trouvée   = {stoch.T12:.2f}")
print("="*50)
# 4. Affiche les graphiques
print(" Affichage des graphiques...")
Graphique.dessiner(phys, stoch)
