import numpy as np
import matplotlib . pyplot as plt
from math import pi
from scipy . stats import multivariate_normal
from scipy . spatial . distance import cdist
from scipy.stats import norm

# 1. Generera och visualisera data

x_vals = np.linspace(-1, 1, 41)  # [-1, -0.95, ..., 0.95, 1] Stegstorlek = 0.05
X1, X2 = np.meshgrid(x_vals, x_vals) # Skapa 2D-grid i input space

w0, w1, w2 = 0, 2.5, -0.5 # Definiera vikter
sigma = 0.3  # Du kan även testa noise-variansen 0.5, 0.8, 1.2

# Beräkna t enligt Eq. 38: t = w0 + w1 * x1^2 + w2 * x2^3 + noise
t_clean = w0 + w1 * (X1 ** 2) + w2 * (X2 ** 3)
noise = np.random.normal(0, sigma, size=X1.shape)
t_noisy = t_clean + noise

# Plotta 3D-yta
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, t_noisy, cmap='viridis', edgecolor='none')
ax.set_title(f"Generated data with noise σ = {sigma}")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("t")
plt.tight_layout()
plt.show()

# 2. Dela upp i tränings- och testdata baserat på x1 och x2
x1_flat = X1.flatten()
x2_flat = X2.flatten()
t_flat = t_noisy.flatten()

# Mask: välj testdata där både |x1| > 0.3 och |x2| > 0.3
test_mask = (np.abs(x1_flat) > 0.3) & (np.abs(x2_flat) > 0.3)
train_mask = ~test_mask  # resten är träningsdata

# Skapa träningsdata
x_train = np.vstack((x1_flat[train_mask], x2_flat[train_mask])).T  # (N_train, 2), Vertikalt stapla dessa två vektorer ovanpå varandra
t_train = t_flat[train_mask]

# Skapa testdata och lägg till extra brus
x_test = np.vstack((x1_flat[test_mask], x2_flat[test_mask])).T  # (N_test, 2), Vertikalt stapla dessa två vektorer ovanpå varandra
t_test = t_flat[test_mask]
sigma_extra = 0.3  # kan justera detta
t_test += np.random.normal(0, sigma_extra, size=t_test.shape)

# 3. ML-regression

# Designfunktion: phi(x) = [1, x1^2, x2^3]
def design_matrix(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return np.vstack((np.ones(len(x1)), x1**2, x2**3)).T  # shape: (N, 3)

# Skapa designmatriser
Phi_train = design_matrix(x_train)
Phi_test = design_matrix(x_test)

w_ML = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t_train # Eq. 19: w_ML = (ΦᵀΦ)^(-1) Φᵀ t

# Eq. 20: β_ML = N / sum of squared residuals
residuals = t_train - Phi_train @ w_ML
sigma2_ML = np.mean(residuals**2)
beta_ML = 1 / sigma2_ML

t_pred = Phi_test @ w_ML # Prediktioner för testdata

# MSE mellan prediktioner och verkliga testvärden
MSE = np.mean((t_pred - t_test)**2)
print(f"ML Mean Squared Error on test data: {MSE:.4f}")