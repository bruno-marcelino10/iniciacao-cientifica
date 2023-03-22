from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def create_plot(n, X_train, y_train):
    tsne = TSNE(n_components=2)
    X_train_novo = tsne.fit_transform(X_train)

    df = pd.DataFrame()
    df["y"] = y_train
    df["PC1"] = X_train_novo[:,0]
    df["PC2"] = X_train_novo[:,1]

    fig = sns.scatterplot(x="PC1", y="PC2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title=f"grafico_{n+1}")    
    
    plt.savefig(f'data/plots/grafico_{n+1}.png')