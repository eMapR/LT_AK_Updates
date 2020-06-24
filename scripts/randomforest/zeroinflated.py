from sklearn import tree

# Make a new class, inherting from the Regressor, overriding fit and predict
# to insert a binary decision tree classifier as a first step to detect zeros
class DecisionTreeZeroInflatedRegressor(tree.DecisionTreeRegressor):

    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 #min_impurity_decrease=0.,
                 min_impurity_split=None,
                 presort=False):
        super(DecisionTreeZeroInflatedRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            presort=presort)
    
        self.dtZeros = tree.DecisionTreeClassifier(
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=None,
            random_state=random_state,
            presort=presort
        )


    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):

        self.dtZeros.fit(X, y!=0,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)

        super(DecisionTreeZeroInflatedRegressor, self).fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        return self


    def predict(self, X, check_input=True):

        P_0 = self.dtZeros.predict(X, check_input=check_input)
        P_1 = super(DecisionTreeZeroInflatedRegressor, self).predict(X, check_input=check_input)
        
        P = P_0*P_1
        
        return P
