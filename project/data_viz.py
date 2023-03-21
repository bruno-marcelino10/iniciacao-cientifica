from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def create_plot(X_train, n):
    tsne = TSNE(n_components=2)
    X_train_novo = tsne.fit_transform(X_train)

    grafico = sns.scatterplot(data=df, x=X_train_novo[0], y=X_train_novo[1])
    plt.savefig(f'grafico_{n}.png')
