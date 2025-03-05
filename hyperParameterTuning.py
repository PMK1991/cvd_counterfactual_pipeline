
from sklearn.model_selection import train_test_split, GridSearchCV

class HyperparameterTuner:
    def __init__(self, model, param_grid, scoring='precision', cv=5, verbose=1, n_jobs=-1):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.grid_search = None

    def tune(self, X, y):
        self.grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, scoring=self.scoring, cv=self.cv, verbose=self.verbose, n_jobs=self.n_jobs)
        self.grid_search.fit(X, y)
        return self.grid_search.best_params_, self.grid_search.best_score_

    def get_best_estimator(self):
        if self.grid_search:
            return self.grid_search.best_estimator_
        else:
            raise ValueError("You need to run the tune method first.")
