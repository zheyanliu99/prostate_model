# %% import package and read data
import pandas as pd 
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/prostate_data.csv')
df_train =  df[df['train']=='T']
X = df_train.drop(['id', 'train', 'lpsa'], axis=1)
y = df_train.lpsa
df_test =  df[df['train']=='F']
X_test = df_test.drop(['id', 'train', 'lpsa'], axis=1)
y_test = df_test.lpsa

# %% a) Best-subset linear regression with k chosen by 5-fold cross-validation. 
def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def getBest(k):
    
    tic = time.time()
    
    results = []
    
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model


models_best = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
for i in range(1,8):
    models_best.loc[i] = getBest(i)

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

# %% Lasso 

## BIC
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline

start_time = time.time()
lasso_lars_ic = make_pipeline(
    StandardScaler(), LassoLarsIC(criterion="bic", normalize=False)
).fit(X, y)
fit_time = time.time() - start_time

results = pd.DataFrame(
    {
        "alphas": lasso_lars_ic[-1].alphas_,
    }
).set_index("alphas")
alpha_aic = lasso_lars_ic[-1].alpha_

results["BIC criterion"] = lasso_lars_ic[-1].criterion_
alpha_bic = lasso_lars_ic[-1].alpha_

def highlight_min(x):
    x_min = x.min()
    return ["font-weight: bold" if v == x_min else "" for v in x]


results.style.apply(highlight_min)

ax = results.plot()
ax.vlines(
    alpha_bic,
    results["BIC criterion"].min(),
    results["BIC criterion"].max(),
    label="alpha: BIC estimate",
    linestyle="--",
    color="tab:orange",
)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("criterion")
ax.set_xscale("log")
ax.legend()
_ = ax.set_title(
    f"Information-criterion for model selection (training time {fit_time:.2f}s)"
)


# %% Lasso cv
## cross validation
from sklearn.linear_model import LassoCV

start_time = time.time()
model = make_pipeline(StandardScaler(), LassoCV(cv=5)).fit(X, y)
fit_time = time.time() - start_time

import matplotlib.pyplot as plt

ymin, ymax = 0, 8
lasso = model[-1]
plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
plt.plot(
    lasso.alphas_,
    lasso.mse_path_.mean(axis=-1),
    color="black",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

plt.ylim(ymin, ymax)
plt.xlabel(r"$\alpha$")
plt.ylabel("Mean square error")
plt.legend()
_ = plt.title(
    f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
)




# %% PCM
scores_mse = []
for k in range(1, 9):
    model = PCA(n_components=k)
    model.fit(X)
    X_train_pca = model.transform(X)
    mse = -cross_val_score(LinearRegression(), X_train_pca, y, 
                           cv=5, scoring='neg_mean_squared_error')
    scores_mse.append(np.mean(mse))
min(scores_mse)
 
index = np.argmin(scores_mse)
index
 
plt.plot(range(1, 9), scores_mse)
plt.axvline(index + 1, color='k', linestyle='--', linewidth=1)
plt.xlabel('Number of Components')
plt.ylabel('Mean Squared Error')
plt.title('5-fold Cross-validation Error')
plt.tight_layout()

# %%
