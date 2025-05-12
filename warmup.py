#import pylab as pb
import numpy as np
import matplotlib . pyplot as plt
from math import pi
from scipy . stats import multivariate_normal
from scipy . spatial . distance import cdist
from scipy.stats import norm
# To draw n samples from multivariate Gaussian distribution with mu and Cov:
# f = np . random . multivariate_normal ( mu , Cov , n )


#TASK 1

# Parametrar
w0, w1 = -1.2, 0.9
sigma2 = 0.8  # testa även 0.1, 0.4, 0.8, Fråga 7
N_train = 100 # testa även 10, 20

# Träningsdata
# Skapa x (träningsinput) som går mellan 1 och -1
x_train = np.linspace(-1, 1, N_train)
# Skapa t (träningsoutput) med fel/brus ε
t_train = w0 + w1 * x_train + np.random.normal(0, np.sqrt(sigma2), size=N_train)

# Testdata
# Skapa x = [-1.5, -1.4, …, -1.1] U [1.1, …, 1.5]
x_test = np.concatenate([
    np.linspace(-1.5, -1.1, 5),
    np.linspace(1.1, 1.5, 5)
])
# Skapa t på samma sätt som tidigare
t_test = w0 + w1 * x_test + np.random.normal(0, np.sqrt(sigma2), size=len(x_test))

# === PLOT ===
w0list = np . linspace ( -3.0 , 2.0 , 200)
w1list = np . linspace ( -2.0 , 4.0 , 200)
W0arr , W1arr = np . meshgrid ( w0list , w1list )
pos = np . dstack (( W0arr , W1arr ))


# 1.  Beräkna prior Eq. 23

alpha = 2
mean_prior = np.array([0,0])
variance_prior = (1/alpha) * np.identity(2)

prior_distribution =  multivariate_normal(mean=mean_prior, cov=variance_prior) # Beräknar normalfördelnigen
prior_pdf = prior_distribution.pdf(pos)

"""
plt . contour ( W0arr , W1arr , prior_pdf )
plt . show ()
"""
# 2.	Beräkna likelihood Eq. 17
X_ext = np.vstack((np.ones_like(x_train), x_train)).T # Skapa designmatris
# Förbered en tom 2D-array som ska fyllas med likelihood-värden
# Storlek matchar rutnätet av w₀ och w₁ (200 × 200)
likelihood = np.zeros_like(W0arr)

# Loopa över alla kombinationer av w₀ och w₁ i rutnätet
for i in range(W0arr.shape[0]):
    for j in range(W0arr.shape[1]):
        w = np.array([W0arr[i, j], W1arr[i, j]]) # w = [w0, w1] från rutnätet
        mu = X_ext @ w # multiplisera matrisen X_ext med vektorn w
        probs = norm.pdf(t_train, loc=mu, scale=np.sqrt(sigma2)) # Beräknar sannolikheten för varje t_n enligt N(t_n | μ_n, σ²), antar oberoende normalfördelningar
        likelihood[i, j] = np.prod(probs)# Total likelihood = produkt av alla individuella sannolikheter


"""
plt.figure(figsize=(6, 5))
plt.contour(W0arr, W1arr, likelihood)
plt.title("Likelihood p(t | w)")
plt.xlabel("w₀")
plt.ylabel("w₁")
plt.grid(True)
plt.show()
"""

# 3.	Beräkna posterior
beta = 1 / sigma2  # precision för likelihood

I = np.identity(2)
S_N_inv = alpha * I + beta * X_ext.T @ X_ext  # Eq. 28, precisionen (inversen av kovarians) för posteriorn
S_N = np.linalg.inv(S_N_inv) # kovariansen S_N för posteriorn

m_N = beta * S_N @ X_ext.T @ t_train          # Eq. 27, medelvärdet för posteriorn

# Skapa posteriorfördelning
posterior_distribution = multivariate_normal(mean=m_N, cov=S_N)
posterior_pdf = posterior_distribution.pdf(pos)

# === PLOT ===
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rader, 3 kolumner

axs[0, 0].contour(W0arr, W1arr, prior_pdf)
axs[0, 0].set_title("Priorfördelning p(w)")
axs[0, 0].set_xlabel("w₀")
axs[0, 0].set_ylabel("w₁")
axs[0, 0].grid(True)

axs[0, 1].contour(W0arr, W1arr, likelihood)
axs[0, 1].set_title("Likelihood p(t | w)")
axs[0, 1].set_xlabel("w₀")
axs[0, 1].set_ylabel("w₁")
axs[0, 1].grid(True)

axs[0, 2].contour(W0arr, W1arr, posterior_pdf)
axs[0, 2].set_title("Posterior p(w | t)")
axs[0, 2].set_xlabel("w₀")
axs[0, 2].set_ylabel("w₁")
axs[0, 2].grid(True)


axs[1, 0].scatter(x_train, t_train, color='black', alpha=0.5, label="Träningsdata")
axs[1, 0].scatter(x_test, t_test, color='red', marker='x', label="Testdata")

axs[1, 0].set_title("Modeller från posteriorn")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4.	Dra model samples från posteriors och rita linjer

# Dra 5 samples från posteriorn
samples = posterior_distribution.rvs(size=5)
x_plot = np.linspace(-1.6, 1.6, 100)

# Rita varje sampled modell i subplot 4
for w_sample in samples:
    w0_sample, w1_sample = w_sample
    y_plot = w0_sample + w1_sample * x_plot
    axs[1, 0].plot(x_plot, y_plot, label=f"w = [{w0_sample:.2f}, {w1_sample:.2f}]")

# 5.	Gör Bayesianska prediktioner med osäkerhet

# Skapa designmatris för testpunkter (phi(x) = [1, x])
X_test_ext = np.vstack((np.ones_like(x_test), x_test)).T

# Prediktionens medelvärde och osäkerhet
mu_test = X_test_ext @ m_N # eq33
sigma2_test = np.array([1 / beta + x.T @ S_N @ x for x in X_test_ext]) #eq34
std_test = np.sqrt(sigma2_test)

# Rita i subplot 5
axs[1, 1].plot(x_test, mu_test, label="Bayesianskt medel", color='blue')
axs[1, 1].fill_between(x_test, mu_test - std_test, mu_test + std_test, alpha=0.3, color='blue', label="±1 std")
axs[1, 1].scatter(x_test, t_test, color='red', marker='x', label="Testdata")
axs[1, 1].scatter(x_train, t_train, color='black', alpha=0.2, label="Träningsdata")

axs[1, 1].set_title("Bayesianska prediktioner")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("t")
axs[1, 1].legend()
axs[1, 1].grid(True)

# 6.	Gör ML-prediktion

# 1. Beräkna ML-vikter
w_ml = np.linalg.inv(X_ext.T @ X_ext) @ X_ext.T @ t_train  # Eq. från sektion 1.3
w0_ml, w1_ml = w_ml

# 2. Skapa x-värden och prediktion med ML
x_plot = np.linspace(-1.6, 1.6, 100)
y_ml = w0_ml + w1_ml * x_plot

axs[1, 2].plot(x_plot, y_ml, label="ML-prediktion", color='green')
axs[1, 2].scatter(x_train, t_train, color='black', alpha=0.3, label="Träningsdata")
axs[1, 2].scatter(x_test, t_test, color='red', marker='x', label="Testdata")

axs[1, 2].set_title("ML-prediktion")
axs[1, 2].set_xlabel("x")
axs[1, 2].set_ylabel("t")
axs[1, 2].legend()
axs[1, 2].grid(True)

plt.tight_layout()
plt.show()



"""	7.	Visualisera och jämför
"""
