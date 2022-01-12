from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import confusion_matrix, make_scorer, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import ParameterGrid, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict
from preprocess import clean_dataframe, to_numeric, bin_age, get_train_test_val

### Constants

random_state = 1
input_features = ['County', 'State', 'Area', 'VicAge', 'VicSex', 'VicRace', 'VicEthnic', 'VicCount', 'Weapon', 'Subcircum', 'Agency', 'Agentype', 'Circumstance', 'Homicide']
input_features_meta = ['County', 'State', 'Area', 'VicAge', 'VicSex', 'VicRace', 'VicEthnic', 'VicCount', 'Weapon', 'Subcircum', 'Agency', 'Agentype', 'Circumstance', 'Homicide',
                        'OffAge_pred', 'OffSex_pred', 'OffRace_pred', 'OffEthnic_pred', 'OffCount_pred', 'Relationship_pred']
output_features = ['OffAge', 'OffSex', 'OffRace', 'OffEthnic', 'OffCount', 'Relationship']
non_numeric_inputs = ['County', 'State', 'Area', 'VicSex', 'VicRace', 'VicEthnic', 'Weapon', 'Subcircum', 'Agency', 'Agentype', 'Circumstance', 'Homicide']
non_numeric_binned = ['County', 'State', 'Area', 'VicAge', 'VicSex', 'VicRace', 'VicEthnic', 'Weapon', 'Subcircum', 'Agency', 'Agentype', 'Circumstance', 'Homicide']
non_numeric_outputs_binned = ['OffAge_pred', 'OffSex_pred', 'OffRace_pred', 'OffEthnic_pred', 'Relationship_pred']
names = {   
    'OffAge': ['0-11','12-14', '15-17', '18-21', '22-24', '25-29','30-34','35-39','40-49','50-64','65+', 'Unknown'],
    'OffSex': ['Male', 'Female'],
    'OffRace': ['White', 'Black', 'Asian', 'American Indian or Alaskan Native', 'Native Hawaiian or Pacific Islander'], 
    'OffEthnic': ['Hispanic origin', 'Not of Hispanic origin', 'Unknown'],
    'OffCount': [x for x in range(11)],
    'Relationship': ['Acquaintance', 'Stranger', 'Unknown', 'Other - known to victim', 'Wife', 'Friend', 'Girlfriend', 'Son',
    'Other family', 'Husband', 'Boyfriend', 'Daughter', 'Neighbor', 'Brother', 'Father', 'Mother', 'In-law', 'Common-law wife',
    'Common-law husband', 'Ex-wife', 'Stepfather', 'Sister', 'Homosexual relationship', 'Stepson', 'Ex-husband', 'Stepdaughter',
    'Employer', 'Employee', 'Stepmother']
    }

# these weights have been found after optimizing the decision trees for the respective variable
optimal_alphas_unweighted = {'OffAge': 0.0007174839999999998, 'OffSex': 0.0, 'OffRace': 0.0005803239999999995, 'OffEthnic': 0.0013096340000000003, 'OffCount': 0.0000121990000000000, 'Relationship': 0.0007309949999999995}
optimal_alphas_balanced_weights = {'OffAge': 0.0008197889999999998, 'OffSex': 0.000529296, 'OffRace': 0.000916389, 'OffEthnic': 0.00027699699999999964, 'OffCount': 0.000098790000000000, 'Relationship': 0.0009406909999999997}
optimal_alphas_meta_unweighted = {'OffAge': 0.0005834199999999995, 'OffSex': 0, 'OffRace': 0.0005803239999999995, 'OffEthnic': 0.0013096340000000003, 'OffCount': 0.0012110000000000003, 'Relationship': 0.004639935}
optimal_alphas_meta_balanced_weights = {'OffAge': 0.0008197889999999998, 'OffSex': 0.0014489100000000003, 'OffRace': 0.0007608829999999995, 'OffEthnic': 0.0002915079999999996, 'OffCount': 0.0002706559999999995, 'Relationship': 0.009452057000000017}


### Methods


def prepare_data(train_encoder: bool = False) -> pd.DataFrame:
    """loads "Murder_Data.zip" file, cleans the data and return stratified samples for training/testing/validation.
        Use of this function should be mututally exclusive with method "load_subsets".

    Args:
        train_encoder (bool, optional): [description]. Defaults to False.

    Returns:
        pd.DataFrame: Three data frames, containing training/testing/validation subsets
    """

    raw_data = pd.read_csv('Murder_Data.zip', index_col=0, compression='zip', low_memory=False)
    cleaned_data = clean_dataframe(raw_data)
    cleaned_data['VicAge'].replace(to_replace='Unknown', value = 999, inplace = True)
    cleaned_data['OffAge'] = bin_age(cleaned_data, 'OffAge')

    # strangly, the 'OffAge' column is shown to be of type Int, event though bin_age() casted it to type string.
    # print(cleaned_data['OffAge'].value_counts())

    # OffCount is exempted from stratificaion to prevent too many combinations
    _, train_sample, test_sample, validation_sample = get_train_test_val(cleaned_data, ['OffAge', 'OffSex', 'OffRace', 'OffEthnic'])

    if train_encoder:
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit(cleaned_data[non_numeric_inputs])
        return train_sample, test_sample, validation_sample, ordinal_encoder

    return train_sample, test_sample, validation_sample


def load_subsets() -> pd.DataFrame:
    """Loads subsets saved in train/test/validation_subset.csv. This is an alternative to loading, cleaning and sampling from "Murder_Data.zip".
        The use of this function should be mutually exclusive with the "prepare_data" method.

    Returns:
        pd.DataFrame: Three data frames, containing training/testing/validation subsets
    """
    
    train_sample = pd.read_csv('train_subset.csv', index_col=0, low_memory=False)
    test_sample = pd.read_csv('test_subset.csv', index_col=0, low_memory=False)
    validation_sample = pd.read_csv('validation_subset.csv', index_col=0, low_memory=False)

    return train_sample, test_sample, validation_sample


def save_subsets_to_csv(train_sample, test_sample, validation_sample) -> None:
    """Saves the three given data frames as .csv files in the current working directory. Naming is fixed.

    Args:
        train_sample (pd.DataFrame): data set for training
        test_sample (pd.DataFrame): data set for testing
        validation_sample (pd.DataFrame): data set for validation
    """

    train_sample.to_csv('train_subset.csv')
    test_sample.to_csv('test_subset.csv')
    validation_sample.to_csv('validation_subset.csv')


def fit_Encoder_on_inputs() -> OrdinalEncoder:
    """cleans the data set saved in "Murder_Data.zip" and fits an OrdinalEncoder on the non-numeric input features

    Returns:
        OrdinalEncoder: ordinal encoder fitted on non-numeric input features
    """

    # fits an OrdinalEncoder on the entire, cleaned data set
    raw_data = pd.read_csv('Murder_Data.zip', index_col=0, compression='zip', low_memory=False)
    cleaned_data = clean_dataframe(raw_data)
    cleaned_data['VicAge'].replace(to_replace='Unknown', value = 999, inplace = True)
    cleaned_data['OffAge'] = bin_age(cleaned_data, 'OffAge')
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(cleaned_data[non_numeric_inputs])

    return ordinal_encoder


def fit_Encoder_on_outputs() -> OrdinalEncoder:
    """cleans the data set saved in "Murder_Data.zip" and fits an OrdinalEncoder on the non-numeric output features.
    This is necessary for the multi target stacking model, as predictions from the standard trees are used as inputs.

    Returns:
        OrdinalEncoder: ordinal encoder fitted on non-numeric input features
    """

    # fits an OrdinalEncoder on the entire, cleaned data set
    raw_data = pd.read_csv('Murder_Data.zip', index_col=0, compression='zip', low_memory=False)
    cleaned_data = clean_dataframe(raw_data)
    cleaned_data['VicAge'].replace(to_replace='Unknown', value = 999, inplace = True)
    cleaned_data['OffAge'] = bin_age(cleaned_data, 'OffAge')
    renaming_dict = {}
    for feature in non_numeric_outputs_binned:
        renaming_dict[feature.replace('_pred', '')] = feature
    cleaned_data.rename(columns=renaming_dict, inplace=True)
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(cleaned_data[non_numeric_outputs_binned])

    return ordinal_encoder


def find_pruning_parameter(input_features: List[str], output_feature: str, train_data:pd.DataFrame, validation_data:pd.DataFrame,
                            criterion='gini', scorer=balanced_accuracy_score, exhaustive_search=False, ccp_alphas=None,
                            class_weights=None, random_state=random_state):
    """finds best cost-complexity pruning parameter for decision tree. If exact solution is computationally too expensive, 
        specify values in ccp_alphas manually, which will then be tested.

    Args:
        input_features (List[str]): names of input features
        output_feature (str): name of output feature
        train_data (pd.DataFrame): data set for training
        validation_data (pd.DataFrame): datas set for validation
        criterion (str, optional): criterion to decide whether or not to perform a split in the tree. Defaults to 'gini'.
        scorer (callable, optional): score function to optimize. Defaults to balanced_accuracy_score.
        exhaustive_search (bool, optional): enable exhaustive search (brute force). Defaults to False.
        ccp_alphas (List, optional): values to test if exhaustive search is disabled. If None, search will be performed to find the best value up to 8 decimal places. Defaults to None.
        class_weights (str, optional): defines if classes should be weighted to reduce bias. Must be either None or 'balanced'. Defaults to None.
        random_state (int, optional): for reproducibility. Defaults to (in "Constants" defined) random_state.

    Returns:
        float: best cost-complexity pruning parameter found during search
    """

    # stop execution if inputs are invalid                         
    assert criterion in ['entropy', 'gini'], 'invalid choice of criterion. Needs to be entropy or gini.'
    assert class_weights in ['balanced', None], 'invalid choice of class_weights. Needs to be balanced or None.'
    if(exhaustive_search & (ccp_alphas!=None)):
        raise ValueError('cannot perform exhaustive search and check for manual values. Either set exhaustive search to False or ccp_alphas to None.')

    if exhaustive_search:
        # generate all possible ccp_alpha values for a given sample
        full_tree = DecisionTreeClassifier(random_state=random_state)
        full_tree.fit(train_data[input_features],train_data[output_features])
        ccp_alphas = full_tree.cost_complexity_pruning_path(train_data[input_features],train_data[output_features])['ccp_alphas']

    # since we work with a pre-defined validation set, we need to pass the combined data (training and validation) with 
    # a list of indices which example is for training and which is for validation to the sklearn GridSearchCV method
    split_index = [-1]*len(train_data[input_features]) + [0]*len(validation_data[input_features])
    X = train_data[input_features].append(validation_data[input_features], ignore_index=True)
    y = train_data[output_feature].append(validation_data[output_feature], ignore_index=True)

    if ccp_alphas is None:
        # starts with a ccp_alpha of 0.01 and then iteratively finds the best value up to 8 decimals of precision
        alpha = 0.01
        magnitude = 0.001

        while magnitude > 1e-9:
            if alpha-magnitude > 0:
                grid = [alpha-magnitude, alpha, alpha+magnitude]
            else:
                # ccp_alpha values of >0 are ill-defined
                grid = [0, alpha, alpha+magnitude]
            grid_search = GridSearchCV(
                                estimator=DecisionTreeClassifier(criterion=criterion, class_weight=class_weights, random_state=random_state),
                                scoring=make_scorer(scorer), 
                                param_grid=ParameterGrid({"ccp_alpha": [[candidate] for candidate in grid]}),
                                n_jobs=-1,
                                cv=PredefinedSplit(split_index)
                            )

            grid_search.fit(X, y)
            if alpha == grid_search.best_params_['ccp_alpha']:
                magnitude = magnitude/10
            else:
                alpha = grid_search.best_params_['ccp_alpha']

    else:
        grid_search = GridSearchCV(
                            estimator=DecisionTreeClassifier(random_state=random_state, class_weight=class_weights),
                            scoring=make_scorer(scorer),
                            param_grid=ParameterGrid({"ccp_alpha": [[alpha] for alpha in ccp_alphas]}),
                            n_jobs=-1,
                            cv=PredefinedSplit(split_index),
                            error_score='raise'
                        )
        grid_search.fit(X, y)
        alpha = grid_search.best_params_['ccp_alpha']

    return alpha


def fit_tree_for_alpha(X_train, y_train, ccp_alpha, class_weights=None, random_state=random_state) -> DecisionTreeClassifier:
    """fits and returns a Decision Tree for the specified pruning parameter ccp_alpha

    Args:
        X_train (pd.DataFrame): input features for training
        y_train (pd.DataFrame): (correct) output features for training
        ccp_alpha (List): cost-complexity pruning parameter
        class_weights (str, optional): defines if classes should be weighted to reduce bias. Must be either None or 'balanced'. Defaults to None.
        random_state (int, optional): for reproducibility. Defaults to (in "Constants" defined) random_state.

    Returns:
        DecisionTreeClassifier: fitted (and potentually pruned) tree
    """

    assert class_weights in ['balanced', None], 'invalid choice of class_weights. Needs to be balanced or None.'

    return DecisionTreeClassifier(ccp_alpha=ccp_alpha, class_weight=class_weights, random_state=random_state).fit(X_train, y_train)


def fit_all_trees(train_data, standard=True, meta_train_data_weighted=None, random_state=random_state) -> List[DecisionTreeClassifier]:
    """fits and returns a Decision Tree for the specified pruning parameter ccp_alpha

    Args:
        train_data (pd.DataFrame): training data. If standard=False, this set will be used for the unweighted meta trees.
        class_weights (str, optional): defines if classes should be weighted to reduce bias. Must be either None or 'balanced'. Defaults to None.
        standard (bool): Defines if standard or meta trees are to be fitted. If false, a secondary training set needs to be specified that is used for the weighted meta trees.
                        The first data set (train_data) will be used to fit the unweighted meta trees. Defaults to true.
        meta_train_data_weighted (pd.DataFrame): training data for weighted meta trees. Defaults to None.
        random_state (int, optional): for reproducibility. Defaults to (in "Constants" defined) random_state.

    Returns:
        List[DecisionTreeClassifier]: two lists of fitted (and potentually pruned) trees for all output variables - the first without weights, the second with weights
    """

    fitted_trees_unweighted = dict.fromkeys(output_features)
    fitted_trees_weighted = dict.fromkeys(output_features)
    
    if standard: 
        input_vars = input_features
        alphas_unweighted = optimal_alphas_unweighted
        alphas_weighted = optimal_alphas_balanced_weights

        for output_var in output_features:
            fitted_trees_unweighted[output_var] = fit_tree_for_alpha(train_data[input_vars], train_data[output_var], ccp_alpha=alphas_unweighted[output_var])
            fitted_trees_weighted[output_var] = fit_tree_for_alpha(train_data[input_vars], train_data[output_var], ccp_alpha=alphas_weighted[output_var], class_weights='balanced')

    else: 
        input_vars = input_features_meta
        alphas_unweighted = optimal_alphas_meta_unweighted
        alphas_weighted = optimal_alphas_meta_balanced_weights

        for output_var in output_features:
            fitted_trees_unweighted[output_var] = fit_tree_for_alpha(train_data[input_vars], train_data[output_var], ccp_alpha=alphas_unweighted[output_var])
            fitted_trees_weighted[output_var] = fit_tree_for_alpha(meta_train_data_weighted[input_vars], meta_train_data_weighted[output_var], ccp_alpha=alphas_weighted[output_var], class_weights='balanced')
        
    return fitted_trees_unweighted, fitted_trees_weighted


def num_correct_predictions(actual_values, predictions) -> List:
    """calculates for each observation the number of (output) variablles that were predicted correctly

    Args:
        actual_values (pd.DataFrame): true values for the predicted variables
        predictions (pd.DataFrame): values predicted by a model

    Returns:
        List: contains the number of correctly predicted variables for each observation
    """

    correctly_predicted = []
    combined = actual_values.join(predictions)

    for index, row in combined.iterrows():
        counter = 0
        for output_var in output_features:
            counter += (int)(row[output_var]==row[output_var + '_pred'])
        correctly_predicted.append(counter)

    return correctly_predicted


def predict_all(to_be_predicted_data, input_feat=None, trained_trees=None, train_data=None, output_features=output_features, standard=True, ccp_alphas=None, class_weights=None, random_state=random_state) -> pd.DataFrame:
    """Fits one tree for each output feature on the training data set and returns their predictions on the test data set

    Args:
        to_be_predicted_data (pd.DataFrame): instances of data for which predictions are to be made.
        input_features (List): names of input features. Defaults to (in "Constants" defined) input_features/input_features_meta.
        trained_trees (Dict): DecisionTreeClassifiers to use for predictions. Keys must be identical to output_features.
                                If trained_trees=None, new trees will be generated and fitted. For this, train_datas needs to be specified. Defaults to None.
        train_data (pd.DataFrame): data to be used for fitting trees, should no trained trees be passed. Defaults to None
        output_features (List): names of output features. Defaults to (in "Constants" defined) output features.
        standard (bool): Defines if predictions are made for standard or meta trees. Defaults to true.
        ccp_alphas (dict, optional): dictionary of cost-complexity pruning parameters for all output features. If ccp_alphas=None, then the in "Constants" defined alphas will be used. Defaults to None.
        class_weights (str, optional): defines if classes should be weighted to reduce bias. Must be either None or 'balanced'. Defaults to None.
        random_state (int, optional): for reproducibility. Defaults to (in "Constants" defined) random_state.

    Returns:
        pd.DataFrame: predictions for all output_features for each observation
    """

    assert class_weights in ['balanced', None], 'invalid choice of class_weights. Needs to be balanced or None.'

    predictions: dict = {}

    if ((class_weights is None) & (ccp_alphas is None)):
        if standard: ccp_alphas=optimal_alphas_unweighted
        else: ccp_alphas=optimal_alphas_meta_unweighted
    elif ((class_weights=='balanced') & (ccp_alphas is None)):
        if standard: ccp_alphas=optimal_alphas_balanced_weights
        else: ccp_alphas=optimal_alphas_meta_balanced_weights

    if standard: 
        addition = '_pred'
        if input_feat is None: input_feat=input_features
    else: 
        addition = '_pred_meta'
        if input_feat is None: input_feat=input_features_meta

    if trained_trees is None:
        if train_data is None:
            raise ValueError('Train_data variable is None, but needs to be specified to train trais. Either specify train_data or trained_trees.')
        trained_trees: dict = dict.fromkeys(output_features)
        for output_var in output_features:
            trained_trees[output_var] = fit_tree_for_alpha(train_data[input_features], train_data[output_var], ccp_alphas[output_var], random_state=random_state, class_weights=class_weights)

    for output_var in output_features:
        predictions[output_var + addition] = trained_trees[output_var].predict(to_be_predicted_data[input_feat])

    return pd.DataFrame(predictions)


#def plot_and_save_tree(tree: DecisionTreeClassifier, y_test: pd.DataFrame, class_names: List, out_file: str = None,):
#    dot_data = tree.export_graphviz(tree, out_file=out_file, class_names=class_names, filled=True)
#    graphviz.Souce(dot_data, format='png')


def generate_data_for_meta_trees(trained_standard_trees, train_data, test_data, validation_data, trained_encoder_outputs, non_numeric_outputs=non_numeric_outputs_binned) -> pd.DataFrame:
    """Uses trained standard trees to make predictions for thee train, test and validation data sets. Data sets are joined with predictions to obtain data sets to use with meta trees.

    Args:
        trained_standard_trees (Dict): DecisionTreeClassifiers to use for predictions. Keys must be identical to output_features.
        train_data (pd.DataFrame): data set for training
        test_data (pd.DataFrame): data set for testing
        validation_data (pd.DataFrame): data set for validation
        trained_encoder_outputs (OrdinalEncoder): OrdinalEncoder that was trained on output features.
        non_numeric_outputs ([type], optional): List of output features that are to be encoded with the Ordinal Encoder. Defaults to non_numeric_outputs_binned.

    Returns:
        pd.DataFrame: Three data frames: one for training, testing and validating meta decision trees.
    """

    predictions_train = predict_all(train_data, trained_trees=trained_standard_trees)
    predictions_test = predict_all(test_data, trained_trees=trained_standard_trees)
    predictions_validation = predict_all(validation_data, trained_trees=trained_standard_trees)

    predictions_train_numeric = to_numeric(predictions_train, ordinal_encoder=trained_encoder_outputs, non_numeric_features=non_numeric_outputs)
    predictions_test_numeric = to_numeric(predictions_test, ordinal_encoder=trained_encoder_outputs, non_numeric_features=non_numeric_outputs)
    predictions_validation_numeric = to_numeric(predictions_validation, ordinal_encoder=trained_encoder_outputs, non_numeric_features=non_numeric_outputs)

    meta_train_data = train_data.join(predictions_train_numeric)
    meta_test_data = test_data.join(predictions_test_numeric)
    meta_validation_data = validation_data.join(predictions_validation_numeric)

    return meta_train_data, meta_test_data, meta_validation_data


def create_meta_trees(input_features, output_features, train_data, test_data, validation_data, ordinal_encoder_outputs, non_numeric_outputs=non_numeric_outputs_binned, ccp_alphas=None, class_weights=None,
                random_state=random_state, return_alphas=False, verbose=True) -> List[DecisionTreeClassifier]:
    """constructs, optimizes and fits a meta decision tree for each output variable. A meta tree is given the original input features plus the prediction of six "standard" decision trees,
        which each predict one output variable

    Args:
        input_featueres (List): names of input features
        output_features (List): names of output features
        train_data (pd.DataFrame): data set for training
        test_data (pd.DataFrame): data set for testing
        validation_data (pd.DataFrame): data set for validation
        ordinal_encoder_outputs (OrdinalEncoder): Ordinal Encoder fitted on (non-numeric) output features
        non_numeric_outputs (List[str]): List of output features that are to be encoded with the Ordinal Encoder. Defaults to (in "Constants" defined) non_numeric_outputs_binned.
        ccp_alphas (List, optional): Input for predict_all method, see the documentation there. Defaults to None.
        class_weights (str, optional): defines if classes should be weighted to reduce bias. Must be either None or 'balanced'. Defaults to None.
        random_state (int, optional): for reproducability. Defaults to random_state.
        return_alphas (bool, optional): defines if optimized cost-complexity parameters for the meta-trees should be returned. Defaults to False.
        verbose (bool, optional): prints method progress into console during runtime. Defaults to True.

    Returns:
        List[DecisionTreeClassifier]: list of six meta decision trees, one for each output feature
    """
    # To-Do: Make meta_trees a dictionary instead of a list
    # To-Do: Make method that uses pre-specified alphas to train the meta trees‚
    meta_trees = []
    meta_ccp_alphas = {}
    predictiction_names = []
    for name in output_features:
        predictiction_names.append(name + '_pred')

    # obtain predictions for all outputs and encode them into numeric features (necessary to use these predictions as inputs for meta trees)
    predictions_training = predict_all(train_data, train_data=train_data, ccp_alphas=ccp_alphas, class_weights=class_weights, random_satate=random_state)
    predictions_validation = predict_all(validation_data, train_data=train_data, ccp_alphas=ccp_alphas, class_weights=class_weights, random_state=random_state)
    
    predictions_training_numeric = to_numeric(predictions_training, ordinal_encoder=ordinal_encoder_outputs, non_numeric_features=non_numeric_outputs)
    predictions_validation_numeric = to_numeric(predictions_validation, ordinal_encoder=ordinal_encoder_outputs, non_numeric_features=non_numeric_outputs)

    # merge predictions with original data sets to use as data for the meta learner
    new_input_features = input_features + predictiction_names
    new_train_data = train_data.join(predictions_training_numeric)
    new_validation_data = validation_data.join(predictions_validation_numeric)


    # find optimal cost-complexity pruning parameter for meta learners, then fit them on previously combined data sets
    for output_feature in output_features:
        if verbose: print(f'Currently working on meta learner for {output_feature}...')
        ccp_alpha = find_pruning_parameter(new_input_features, output_feature, new_train_data, new_validation_data,
                                            class_weights=class_weights, random_state=random_state)
        meta_ccp_alphas[output_feature] = ccp_alpha
        meta_trees.append(fit_tree_for_alpha(new_train_data[new_input_features], train_data[output_feature], ccp_alpha,
                                                class_weights, random_state))

    if return_alphas:
        return meta_trees, meta_ccp_alphas
    else:
        return meta_trees


        
def feature_importance_heatmap(trained_trees) -> ff:
    """generates a plotly heatmap to visualize the feature importance metric of decision trees

    Args:
        trained_tree (dict): keys are names of target varaibles, values are trained decision trees to extract feature importances from

    Returns:
        ff.figure_factory: plotly heatmap
    """
    feature_importances = []

    for var in output_features:
        feature_importances.append(trained_trees[var].feature_importances_.tolist())

    numpy_importances = np.array(feature_importances)
    numpy_importances = numpy_importances.transpose()
    importances_rounded = np.around(numpy_importances, decimals=4)

    fig = ff.create_annotated_heatmap(numpy_importances, 
                                        x=output_features, 
                                        y=input_features,
                                        annotation_text=importances_rounded)
    fig.update_xaxes(side="bottom", tickangle=-45)
    fig.update_layout(title_text='Feature Importances',
                        autosize=False,
                        width=1000,
                        height=1000,
                        showlegend=True,
                        font={'size': 26}
                        )
    fig['data'][0]['showscale'] = True
    return fig


def plotly_confusion_matrix(actual_values, predicted_values, labels) -> ff:
    # unfinished
    cm = confusion_matrix(actual_values, predicted_values, labels=labels)
    print('Confusion matrix:')
    print(cm)
    cm = cm[::-1]
    fig = ff.create_annotated_heatmap(z=cm, 
                                        x=labels.tolist(), 
                                        y=labels[::-1].copy().tolist(),
                                        annotation_text=cm)

    #fig = px.imshow(cm, x=labels, y=labels, aspect='equal')
    #fig.update_traces(text=cm)

    fig.update_xaxes(side="bottom", tickangle=-45)
    fig.update_layout(title_text='Confusion matrix',
                        autosize=False,
                        width=1000,
                        height=1000,
                        font={'size': 26},
                        xaxis=dict(scaleanchor='y',constrain='domain')
                        )

    return fig


def find_majority_minority_classes(value_names:List, occurances:List):
    # majority class is a class that has at least 25% more observations than if occurances were uniformly distributed
    # minority class is a class that has at least 25% less observations than if occurances were uniformly distributed
    if len(value_names) != len(occurances):
        raise(ValueError(f'Length of class_names and occurances dont match. Length of class_names is {len(value_names)}, length of occurances is {len(occurances)}'))
    sum_occurances = sum(occurances)
    majorit_threshold = (sum_occurances/len(occurances))*1.25
    minority_threshold = (sum_occurances/len(occurances))*0.75
    class_balances = {}
    for index in range(len(value_names)):
        if occurances[index] > majorit_threshold:
            class_balances[value_names[index]] = 'majority'
        elif occurances[index] < minority_threshold:
            class_balances[value_names[index]] = 'minority'
        else: class_balances[value_names[index]] = 'average'
    return class_balances


def plot_metrics(tree_names:List, tree_reports:List, predicted_feature:str):
    # method that plots the metrics (either precision, recall or F1) for every value of a variable
    # as a number of crosses on a vertical line; this is done for different types of trees that are then put besides each other
    # https://plotly.com/python/dot-plots/
    # unfinished

    # needed: precision/recall values for each tree
    # tree names
    # values of output variable
    # majority/minority class identifiers

    value_names = names[predicted_feature]
    recall_scores = dict.fromkeys(value_names)
    precision_scores = dict.fromkeys(value_names)
    f1_scores = dict.fromkeys(value_names)
    
    for name in value_names:
        recall = []
        precision = []
        f1 = []
        for report in tree_reports:
            recall.append(report[name]['recall'])
            precision.append(report[name]['precision'])
            f1.append(report[name]['f1-score'])

        recall_scores[name] = recall
        precision_scores[name] = precision
        f1_scores[name] = f1_scores


    markers = ['x', 'star', 'circle', 'triangle-up', 'hexagon', 'hourglass', 'diamond', 'square', 'cross', 'hash', 'asterisk',
                'x-open', 'star-open', 'circle-open', 'triangle-up-open', 'hexagon-open', 'hourglass-open', 'diamond-open', 'square-open',
                'cross-open', 'hash-open', 'asterisk-open', 'triangle-down', 'triangle-down-open', 'x-dot', 'x-open-dot', 'triangle-left',
                'triangle-left-open', 'triangle-right', 'triangle-right-open']
    
    # class_balance = find_majority_minority_classes()

    fig = go.Figure()
    for i, name in enumerate(value_names):
        fig.add_trace(go.Scatter(
            x=tree_names,
            y=recall_scores[name],
            marker=dict(color="crimson", size=10, symbol=markers[i]),
            mode="markers",
            name=name,
        )) 

    fig.update_layout(title="Comparing trees",
                  xaxis_title="Trees",
                  yaxis_title="score values")

    return fig