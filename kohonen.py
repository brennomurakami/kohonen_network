import numpy as np
import matplotlib.pyplot as plt
import time

# Carregar o conjunto de dados Iris manualmente
# Comprimento da Sépala (Sepal Length)
# Largura da Sépala (Sepal Width)
# Comprimento da Pétala (Petal Length)
# Largura da Pétala (Petal Width)
data = np.array([
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2],
    [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
    [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2],
    [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5],
    [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4],
    [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
    [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [5.2, 4.1, 1.5, 0.1],
    [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.2], [5.0, 3.2, 1.2, 0.2],
    [5.5, 3.5, 1.3, 0.2], [4.9, 3.6, 1.4, 0.1], [4.4, 3.0, 1.3, 0.2],
    [5.1, 3.4, 1.5, 0.2], [5.0, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3],
    [4.4, 3.2, 1.3, 0.2], [5.0, 3.5, 1.6, 0.6], [5.1, 3.8, 1.9, 0.4],
    [4.8, 3.0, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
    [5.3, 3.7, 1.5, 0.2], [5.0, 3.3, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6],
    [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4],
    [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5], [6.0, 2.2, 4.0, 1.0],
    [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
    [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5],
    [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4.0, 1.3],
    [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3],
    [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3.0, 5.0, 1.7],
    [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0], [5.5, 2.4, 3.8, 1.1],
    [5.5, 2.4, 3.7, 1.0], [5.8, 2.7, 3.9, 1.2], [6.0, 2.7, 5.1, 1.6],
    [5.4, 3.0, 4.5, 1.5], [6.0, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
    [6.3, 2.3, 4.4, 1.3], [5.6, 3.0, 4.1, 1.3], [5.5, 2.5, 4.0, 1.3],
    [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4], [5.8, 2.6, 4.0, 1.2],
    [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3], [5.7, 3.0, 4.2, 1.2],
    [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3.0, 1.1],
    [5.7, 2.8, 4.1, 1.3], [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2],
    [7.6, 3.0, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
    [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0],
    [6.4, 2.7, 5.3, 1.9], [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0],
    [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8],
    [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5],
    [6.9, 3.2, 5.7, 2.3], [5.6, 2.8, 4.9, 2.0], [7.7, 2.8, 6.7, 2.0],
    [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6.0, 1.8],
    [6.2, 2.8, 4.8, 1.8], [6.1, 3.0, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
    [7.2, 3.0, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2.0],
    [6.4, 2.8, 5.6, 2.2], [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4],
    [7.7, 3.0, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4], [6.4, 3.1, 5.5, 1.8],
    [6.0, 3.0, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
    [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3],
    [6.7, 3.3, 5.7, 2.5], [6.7, 3.0, 5.2, 2.3], [6.3, 2.5, 5.0, 1.9],
    [6.5, 3.0, 5.2, 2.0], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]
])

# Definindo as classes (apenas para visualização)
target = np.array([0]*50 + [1]*50 + [2]*50)

# Normalizar os dados manualmente
data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Parâmetros da SOM
grid_size = 10  # Tamanho da grade (10x10)
input_dim = data.shape[1]  # Dimensão das entradas (4 características)
initial_lr = 0.1
initial_sigma = max(grid_size, grid_size) / 2
max_epochs = 100

# Inicialização dos pesos aleatórios para cada neurônio na grade
np.random.seed(42)  # Para reprodutibilidade
weights = np.random.random((grid_size, grid_size, input_dim))


# Função para calcular a distância Euclidiana
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# Função para encontrar o neurônio vencedor (BMU)
def find_bmu(sample, weights):
    min_dist = float('inf')
    bmu_idx = (0, 0)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            dist = euclidean_distance(sample, weights[i, j])
            if dist < min_dist:
                min_dist = dist
                bmu_idx = (i, j)
    return bmu_idx


# Função para decair a taxa de aprendizado
def decay_learning_rate(initial_lr, epoch, max_epochs):
    return initial_lr * (1 - epoch / max_epochs)


# Função para decair o sigma
def decay_sigma(initial_sigma, epoch, max_epochs):
    return initial_sigma * (1 - epoch / max_epochs)


# Função para atualizar os pesos dos neurônios
def update_weights(weights, sample, bmu_idx, learning_rate, sigma):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            dist_to_bmu = euclidean_distance(np.array([i, j]), np.array(bmu_idx))
            if dist_to_bmu <= sigma:
                influence = np.exp(-(dist_to_bmu**2) / (2 * (sigma**2)))
                weights[i, j] += learning_rate * influence * (sample - weights[i, j])


# Função para calcular a Quantização de Vetor de Erros (QVE)
def quantization_error(data, weights):
    total_error = 0
    for sample in data:
        bmu_idx = find_bmu(sample, weights)
        total_error += euclidean_distance(sample, weights[bmu_idx[0], bmu_idx[1]])
    return total_error / len(data)


# Treinamento da SOM
start_time = time.time()
for epoch in range(max_epochs):
    for sample in data:
        bmu_idx = find_bmu(sample, weights)
        lr = decay_learning_rate(initial_lr, epoch, max_epochs)
        sigma = decay_sigma(initial_sigma, epoch, max_epochs)
        update_weights(weights, sample, bmu_idx, lr, sigma)

end_time = time.time()  # Registra o tempo final
training_time = end_time - start_time  # Calcula o tempo total de treinamento

# Cálculo da QVE
qve = quantization_error(data, weights)
qve_percentage = (1 - qve / np.max(data)) * 100  # Convertendo para porcentagem

print("Quantização de Vetor de Erros (QVE): {:.2f}%".format(qve_percentage))
print("Tempo total de treinamento: {:.2f} segundos".format(training_time))

# Criação de um mapa de distância para visualização
distance_map = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        dist_sum = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if 0 <= i + di < grid_size and 0 <= j + dj < grid_size:
                    dist_sum += euclidean_distance(weights[i, j], weights[i + di, j + dj])
        distance_map[i, j] = dist_sum

# Visualização dos Resultados
plt.imshow(distance_map, cmap='bone_r')
plt.colorbar()
plt.title('Mapa de Distância')
plt.xlabel('Índice da Coluna')
plt.ylabel('Índice da Linha')

# Marcar as amostras no mapa
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

for cnt, sample in enumerate(data):
    bmu_idx = find_bmu(sample, weights)
    plt.plot(bmu_idx[1] + 0.5, bmu_idx[0] + 0.5, markers[target[cnt]], markerfacecolor='None',
             markeredgecolor=colors[target[cnt]], markersize=12, markeredgewidth=2)

plt.show()

plt.figure(figsize=(10, 6))
for cnt, sample in enumerate(data):
    bmu_idx = find_bmu(sample, weights)
    plt.scatter(sample[0], sample[1], marker=markers[target[cnt]], color=colors[target[cnt]], edgecolors='black', s=100)
    plt.text(sample[0], sample[1], str(bmu_idx), fontsize=8, ha='center', va='center')
plt.title('Scatter Plot das Amostras no Espaço de Características')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.grid(True)
plt.show()