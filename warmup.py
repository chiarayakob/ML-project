#import pylab as pb
import numpy as np
import matplotlib . pyplot as plt
from math import pi
from scipy . stats import multivariate_normal
from scipy . spatial . distance import cdist
# To draw n samples from multivariate Gaussian distribution with mu and Cov:
#f = np . random . multivariate_normal ( mu , Cov , n )


#TASK 1

# Parametrar
w0, w1 = -1.2, 0.9
sigma2 = 0.2  # testa även 0.1, 0.4, 0.8, Fråga 7
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

# 1.  Beräkna prior

alpha = 2
mean_prior = np.array([0,0])
variance_prior = (1/alpha) * np.identity(2)

prior_distribution =  multivariate_normal(mean=mean_prior, cov=variance_prior)

# Plot
w0list = np . linspace ( -3.0 , 2.0 , 200)
w1list = np . linspace ( -2.0 , 2.0 , 200)
W0arr , W1arr = np . meshgrid ( w0list , w1list )
pos = np . dstack (( W0arr , W1arr ))

prior_pdf = prior_distribution.pdf(pos)

plt . contour ( W0arr , W1arr , prior_pdf )
plt . show ()

# 2.	Beräkna likelihood

mean_likelihood = np.array([w0,w1])

likelihood =  multivariate_normal(mean = mean_likelihood, cov = sigma2)
likelihood_pdf = likelihood (pos)
plt . contour ( W0arr , W1arr , likelihood_pdf )
plt . show ()

"""	3.	Beräkna posterior
	4.	Dra samples från posteriors och rita linjer
	5.	Gör Bayesianska prediktioner med osäkerhet
	6.	Gör ML-prediktion
	7.	Visualisera och jämför
"""
