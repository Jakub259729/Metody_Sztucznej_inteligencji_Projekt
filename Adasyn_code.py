import numpy as np
from sklearn import neighbors
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def adasyn(X, y, beta, K, threshold=1):

    #Obliczanie liczby przykładów w klasie mniejszościowej (minor_class) oraz w klasie większościowej(major_class)
    minor_class = int(sum(y))
    major_class = len(y) - minor_class

    #Klasyfikator k-najbliższych sąsiadów musi być dopasowany do danych
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, y)

    #Sprawdzenie czy mamy wystarczająco niezbalansowane dane
    d = np.divide(minor_class, major_class)
    if d > threshold:
        return print("Zbyt male niezbalansowanie")

    #Ile przykładów do wygeerowania
    G = (major_class - minor_class) * beta

    #Obliczenie ilości sąsiadów klasy mniejszościowej dla danego przykładu klasy mniejszościowej.
    Ri = []
    Minority_per_xi = []
    for i in range(minor_class):
        xi = X[i, :].reshape(1, -1)
        neighbours = clf.kneighbors(xi, K+1, False)[0][1:]
        count = 0
        for value in neighbours:
            if y[value] != y[i]:
                count += 1
        Ri.append(count / K)
        minority = []
        for value in neighbours:
            if y[value] != y[i]:
                minority.append(value)
        Minority_per_xi.append(minority)

    #Znormalizowanie Ri aby określić wagę przykładu klasy mniejszościowej przy generowaniu sztucznych przykładów.
    Rhat_i = []
    for ri in Ri:
        rhat_i = ri / sum(Ri)
        Rhat_i.append(rhat_i)
    assert(sum(Rhat_i) > 0.99)

    #Obliczenie ilości danych do wygenerowania dla każdego przykładu klasy mniejszościowej
    Gi = []
    for rhat_i in Rhat_i:
        gi = round(rhat_i * G)
        Gi.append(int(gi))

    #Generowanie przykładów syntetycznych
    syn_data = []
    for i in range(minor_class):
        xi = X[i, :].reshape(1, -1)
        for j in range(Gi[i]):
            if Minority_per_xi[i]:
                index = np.random.choice(Minority_per_xi[i])
                xzi = X[index, :].reshape(1, -1)
                si = xi + (xzi - xi) * np.random.uniform(0, 1)
                syn_data.append(si)

    #Łączenie danych
    X_resampled = np.concatenate((X, np.concatenate(syn_data, axis=0)), axis=0)
    y_resampled = np.concatenate((y, np.ones(len(syn_data))), axis=0)

    return X_resampled, y_resampled

#Generowanie danych
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.7, 0.3], n_informative=5, n_redundant=0, flip_y=0, n_features=10, n_clusters_per_class=7, n_samples=1000)

#Oryginalne wartości
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.5)
plt.title('Oryginalne dane')
plt.show()

#Wykonanie ADASYN
X_resampled, y_resampled = adasyn(X, y, beta=0.9, K=3)

#Dane po wygenerowaniu ADASYN
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, cmap='bwr', alpha=0.5)
plt.title('Dane po wykonaniu ADASYN')
plt.show()