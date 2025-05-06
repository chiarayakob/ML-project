import pylab as pb
import numpy as np
import matplotlib . pyplot as plt
from math import pi
from scipy . stats import multivariate_normal
from scipy . spatial . distance import cdist
# To draw n samples from multivariate Gaussian distribution with mu and Cov:
f = np . random . multivariate_normal ( mu , Cov , n )


#TASK 1

# Träningsdata
# Parametrar
w0, w1 = -1.2, 0.9
sigma2 = 0.2  # testa även 0.1, 0.4, 0.8, Fråga 7
N_train = 100 # testa även 10, 20, Fråga 2

# Skapa x (träningsinput) som går mellan 1 och -1
x_train = np.linspace(-1, 1, N_train)
# Skapa t (träningsoutput) med fel/brus ε
t_train = w0 + w1 * x_train + np.random.normal(0, np.sqrt(sigma2), size=N_train)

""" 1. Beräkna prior
	2.	Beräkna likelihood
	3.	Beräkna posterior
	4.	Dra samples från posteriors och rita linjer
	5.	Gör Bayesianska prediktioner med osäkerhet
	6.	Gör ML-prediktion
	7.	Visualisera och jämför
"""