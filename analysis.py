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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

df = pd.read_csv('data/prostate_data.csv')
df_train =  df[df['train']=='T']
X = df_train.drop(['id', 'train', 'lpsa'], axis=1)
y = df_train.lpsa
df_test =  df[df['train']=='F']
X_test = df_test.drop(['id', 'train', 'lpsa'], axis=1)
y_test = df_test.lpsa

# %% Best-Set cv
# plot
results = []
for k in range(1, 9):
    for combo in itertools.combinations(X.columns, k):
        X_train_bs = X[list(combo)]
        mse = -cross_val_score(LinearRegression(), X_train_bs, y, 
                                cv=5, scoring='neg_mean_squared_error')
        aresult = {'k':k, 'combo': list(combo), 'mse':np.mean(mse), 'mse_path':mse}
        results.append(aresult)

results_df = pd.DataFrame(results)
best_set = results_df.loc[results_df['mse'].argmin()]
results_df_groupbyt = results_df.groupby('k', as_index=False).min('mse')

results_df_groupby = results_df_groupbyt.merge(results_df, on=['k', 'mse'], how='left')

res_arr = results_df_groupby.mse_path[0].reshape(1,-1)
for arr in results_df_groupby.mse_path[1:]:
    res_arr = np.concatenate((res_arr, np.array(arr).reshape(1,-1)), axis=0)

plt.semilogx(results_df_groupby.k, res_arr, linestyle=":")
plt.plot(
    results_df_groupby.k,
    results_df_groupby.mse,
    color="black",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(best_set['k'], linestyle="--", color="black", label="k: CV estimate")

ymin = 0
ymax = 5
plt.ylim(ymin, ymax)
plt.xlabel("k")
plt.ylabel("Mean square error")
plt.legend()
_ = plt.title(
    f"Mean square error on each fold: coordinate descent"
)

# %% Best-Set cv
reg = LinearRegression().fit(X[best_set['combo']], y)
# estimation
cdf = pd.DataFrame(reg.coef_, X[best_set['combo']].columns, columns=['Coefficients'])
print(cdf)
# %% Best-Set cv
# prediction error 
y_pred = reg.predict(X_test[best_set['combo']])
mean_squared_error(y_test, y_pred)
# %% Best-Set cv
# std
np.std(y_pred)/np.sqrt(len(y_pred))

# %% Best-Set BIC
results = []
for k in range(1, 9):
    for combo in itertools.combinations(X.columns, k):
        X_train_bs = X[list(combo)]
        regr = OLS(y, add_constant(X_train_bs)).fit()
        aresult = {'k':k, 'combo':list(combo), 'BIC':regr.bic}
        results.append(aresult)

results_df = pd.DataFrame(results)
best_set = results_df.loc[results_df['BIC'].argmin()]
results_df_groupby = results_df.groupby('k', as_index=False).min('BIC')

ax = results_df_groupby.plot(x='k',y='BIC')
ax.vlines(
    best_set['k'],
    results_df_groupby["BIC"].min(),
    results_df_groupby["BIC"].max(),
    label="k: BIC estimate",
    linestyle="--",
    color="tab:orange",
)
ax.set_xlabel("k")
ax.set_ylabel("BIC criterion")
ax.legend()
_ = ax.set_title(
    f"Information-criterion for model selection"
)
# %% Best-Set cv
reg = LinearRegression().fit(X[best_set['combo']], y)
# estimation
cdf = pd.DataFrame(reg.coef_, X[best_set['combo']].columns, columns=['Coefficients'])
print(cdf)
# %% Best-Set cv
# prediction error 
y_pred = reg.predict(X_test[best_set['combo']])
mean_squared_error(y_test, y_pred)
# %% Best-Set cv
# std
np.std(y_pred)/np.sqrt(len(y_pred))
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
ax.set_ylabel("BIC criterion")
ax.set_xscale("log")
ax.legend()
_ = ax.set_title(
    f"Information-criterion for model selection"
)

# %% Lasso bic
reg = Lasso(alpha=alpha_bic).fit(X, y)
# estimation
cdf = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficients'])
print(cdf)
# %% Lasso bic
# prediction error 
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)
# %% Lasso bic
# std
np.std(y_pred)/np.sqrt(len(y_pred))


# %% Lasso cv
## cross validation
from sklearn.linear_model import LassoCV

start_time = time.time()
model = make_pipeline(StandardScaler(), LassoCV(cv=5)).fit(X, y)
fit_time = time.time() - start_time

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
    f"Mean square error on each fold"
)

# %% Lasso cv
reg = Lasso(alpha=lasso.alpha_).fit(X, y)
# estimation
cdf = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficients'])
print(cdf)
# %% Lasso cv
# prediction error 
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)
# %% Lasso cv
# std
np.std(y_pred)/np.sqrt(len(y_pred))


# %% PCM
## PCM cross validation
scores_mse = []
for k in range(1, 9):
    model = PCA(n_components=k)
    model.fit(X)
    X_train_pca = model.transform(X)
    mse = -cross_val_score(LinearRegression(), X_train_pca, y, 
                           cv=5, scoring='neg_mean_squared_error')
    scores_mse.append(np.mean(mse))
 
index = np.argmin(scores_mse)
 
plt.plot(range(1, 9), scores_mse)
plt.axvline(index + 1, color='k', linestyle='--', linewidth=1)
plt.xlabel('Number of Components')
plt.ylabel('Mean Squared Error')
plt.title('5-fold Cross-validation Error')
plt.tight_layout()

# %% PCM cv
pca = PCA(n_components=index+1)
pca.fit(X)
X_pca = pca.transform(X)
X_test_pca = pca.transform(X_test)
reg = Lasso(alpha=lasso.alpha_).fit(X_pca, y)
# estimation
cdf = pd.DataFrame(reg.coef_, ['PC' + str(i+1) for i in range(index+1)], columns=['Coefficients'])
print(cdf)
# %% PCM cv
# prediction error 
y_pred = reg.predict(X_test_pca)
mean_squared_error(y_test, y_pred)
# %% PCM cv
# std
np.std(y_pred)/np.sqrt(len(y_pred))
