import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
from joblib import load, dump
import numpy as np
import pandas as pd
from skorch import NeuralNetRegressor
from matplotlib import rcParams
from sklearn.inspection import permutation_importance



class configuration:
    experiment: int = 4


DIR = os.path.dirname(__file__)


class NeuralNet_Regressor(NeuralNetRegressor):
    def __init__(
            self,
            module,
            *args,
            criterion=torch.nn.MSELoss,
            **kwargs
    ):
        super(NeuralNetRegressor, self).__init__(
            module,
            *args,
            criterion=criterion,
            **kwargs
        )

    def fit(self, X, y, **fit_params):
        if isinstance(y, np.ndarray) and y.ndim == 1:
            y = y.reshape(-1, 1).astype(np.float32)
        return super(NeuralNet_Regressor, self).fit(X, y, **fit_params) 


def dataset_loader(target):
    p = Path(__file__).resolve()
    p = p.parents[2]
    #url = p.joinpath('data','fulldataset.csv')
    url = p.joinpath('data','osm_nexus_dataset.csv')
    df = pd.read_csv(url,index_col=0)
    # 1.) Targets

    y = df[target]

    # 2.) Drop all non Feature columns from dataset
    X = df.drop(['car_or_van_or_suv_or_pickup','motorcycle_or_bicycle','pedestrian','truck_or_bus','car_or_van_or_suv_or_pickup_moving'], axis = 1)
    

    #Return Features and Targets
    return X, y


def train_test_loader(target, large_r: bool =False):
    from sklearn.model_selection import GroupShuffleSplit
    p = Path(__file__).resolve()
    p = p.parents[2]
    url = p.joinpath('data','osm_nexus_dataset.csv')
    if large_r:
        url = p.joinpath('data','osm_nexus_large_radius.csv')
    df = pd.read_csv(url,index_col=0)
    

    # 2.) Train and Testsplit based on suburbs of the gps locations
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.10, n_splits=2, random_state = 42).split(df, groups=df['suburb']))
    
    #3.) Create train and test
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]

    #4.) Split targets and features. For Bikes and pedestrian temperature and precipitation will also be used as a feature.
    
    if target == 'motorcycle_or_bicycle' or target == 'pedestrian':
       
        X_train = train.drop(['car_or_van_or_suv_or_pickup','motorcycle_or_bicycle','pedestrian',
        'truck_or_bus','car_or_van_or_suv_or_pickup_moving', 'lat', 'lon',"suburb", "timestamp"], axis = 1)
        X_test = test.drop(['car_or_van_or_suv_or_pickup','motorcycle_or_bicycle','pedestrian',
        'truck_or_bus','car_or_van_or_suv_or_pickup_moving', 'lat', 'lon',"suburb","timestamp"], axis = 1)
    else:
        X_train = train.drop(['car_or_van_or_suv_or_pickup','motorcycle_or_bicycle','pedestrian',
        'truck_or_bus','car_or_van_or_suv_or_pickup_moving', 'lat', 'lon',"suburb", "timestamp",'temperature','precipitation'], axis = 1)
        X_test = test.drop(['car_or_van_or_suv_or_pickup','motorcycle_or_bicycle','pedestrian',
        'truck_or_bus','car_or_van_or_suv_or_pickup_moving', 'lat', 'lon',"suburb","timestamp", 'temperature','precipitation'], axis = 1)

    y_test = test[target]
    y_train = train[target]
    train_groups = train["suburb"]
    test_groups = test["suburb"]
    #5.) Return split
    return X_train, X_test , y_train , y_test, train_groups, test_groups

def get_train_test_lat_lon():
    from sklearn.model_selection import GroupShuffleSplit
    p = Path(__file__).resolve()
    p = p.parents[2]
    url = p.joinpath('data','osm_nexus_dataset.csv')
    df = pd.read_csv(url,index_col=0)
    

    # 2.) Train and Testsplit based on suburbs of the gps locations
    train_inds, test_inds = next(GroupShuffleSplit(test_size=.10, n_splits=2, random_state = 42).split(df, groups=df['suburb']))
    
    #3.) Create train and test
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]

    #4 .) Create Dataframe for Train coordinates
    gps_coords_train = pd.DataFrame()
    gps_coords_train['lat'] = train["lat"]
    gps_coords_train['lon'] = train['lon']

    #5 .) Create Dataframe for Test coordinates
    gps_coords_test = pd.DataFrame()
    gps_coords_test['lat'] = test["lat"]
    gps_coords_test['lon'] = test['lon']

    #5.) Return split
    return gps_coords_test, gps_coords_train

def raw_data_loader():
    p = Path(__file__).resolve()
    p = p.parents[2]
    url = p.joinpath('data','fulldataset_50m_sampling.csv')
    df = pd.read_csv(url,index_col=0)
    return df

class Evaluation_Framework():
        def __init__(self, model : str, experiment_number : int = 4):
            self.dir = DIR
            self.targets = ['car_or_van_or_suv_or_pickup','motorcycle_or_bicycle','pedestrian','truck_or_bus','car_or_van_or_suv_or_pickup_moving']
            self.experiment_number = experiment_number
            self.model = model
            if self.model in ["LGB", "lg" , "LGBoost" , "lgb" , "lg_boost", "lightgbm" , "LightGBM" , "boost"]:
                self.modelname = "lg_boost"
                self.filename = "lightgbm_model"
                self.name = "LightGBM"
                self.studyname = "LGB"
            elif self.model in ["RandomForest" , "Random Forest" , "random forest" , "rf" , "RF" ,"random_forest"] :
                self.modelname = "random_forest"
                self.filename = "random_forest_model"
                self.studyname = "RF"
                self.name = "Random forest"
            elif self.model in [ "nn" , "neural_network" , "neuralnetwork" , "neural network"] :
                self.modelname = "neural_network"
                self.filename = "nn_model"
                self.name = "Neural network"
                self.studyname = "nn"
            elif self.model in [ "lr" , "LR" , "linear regression" , "linearregression", "linear"] :
                self.modelname = "lr"
                self.filename = "LR"
                self.name = "LR"
                self.studyname = "LR"
            else:
                raise ValueError("No Model found for that name. Please enter correct modelname")
            

            self.rf_dir = os.path.join(self.dir, "random_forest", 'experiment_{}'.format(self.experiment_number))
            self.lgb_dir = os.path.join(self.dir, "lg_boost", 'experiment_{}'.format(self.experiment_number))


            self.modeldir = os.path.join(self.dir, self.modelname, 'experiment_{}'.format(self.experiment_number))



        def plot_feature_importance(self,target, large_r = False):
            import matplotlib.pyplot as plt
            if self.modelname == "neural_network":
                raise ValueError("Neural Networks don't have feature importance")
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target, large_r=large_r)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target))) 
            features = reg.named_steps["cor_removal"]
            cols = features.get_support(indices=True)
            X_new = X_test.iloc[:, cols]
            features = reg.named_steps["rfe_feature_selection"]
            cols = features.get_support(indices=True)
            X_new = X_new.iloc[:, cols]
            if self.modelname == "lg_boost":
                feat_importances = pd.Series(reg.named_steps["reg"].booster_.feature_importance(importance_type='gain'), index=X_new.columns)
            else:
                feat_importances = pd.Series(reg.named_steps["reg"].feature_importances_, index=X_new.columns)
            plt.figure()
            plt.style.use('ggplot')
            feat_importances.nlargest(n=20).plot(kind='barh', title = 'Feature Importance for {} '.format(target)).invert_yaxis()
            plt.yticks(fontsize= 15)
            plt.title(label = 'Feature Importance {} '.format(target), fontdict={'fontsize' : 18}, pad= 20)
            plt.subplots_adjust(
                top=0.883,
                bottom=0.24,
                left=0.321,
                right=0.764,
                hspace=0.2,
                wspace=0.2
            )
            plt.show()
            plt.close()

        def feature_importance_shap_treemodel(self, target):
            import matplotlib.pyplot as plt
            import shap
            if self.modelname == "neural_network" or self.modelname == "lr":
                raise ValueError("Your selected model is not a tree model")
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target))) 
            
            
            # First feature removal step from pipeline
            features = reg.named_steps["cor_removal"]
            cols = features.get_support(indices=True)
            X_new = X_train.iloc[:, cols]
            # Second feature removal step from pipeline
            features = reg.named_steps["rfe_feature_selection"]
            cols = features.get_support(indices=True)
            X_new = X_new.iloc[:, cols]

            explainer = shap.Explainer(reg.named_steps["reg"])
            shap_values = explainer(X_new)
            f = plt.figure()
            plt.style.use('ggplot')
            plt.rcParams.update({'font.size': 16})
            plt.subplots_adjust(
                top=0.883,
                bottom=0.24,
                left=0.22,
                right=0.7,
                hspace=0.2,
                wspace=0.2
            )
            plt.title("SHAP Values for {}".format(target), pad = 20)
            shap.plots.bar(shap_values, max_display=20)
            plt.show()

        def plot_shap_any_model(self, target):
            import shap
            import matplotlib.pyplot as plt
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target))) 
            explainer = shap.Explainer(reg.named_steps["reg"])
            features = reg.named_steps["cor_removal"]
            cols = features.get_support(indices=True)
            X_new_train = X_train.iloc[:, cols]
            X_new_test = X_test.iloc[:, cols]
            features = reg.named_steps["rfe_feature_selection"]
            cols = features.get_support(indices=True)
            X_new_train = X_new_train.iloc[:, cols]
            X_new_test = X_new_test.iloc[:, cols]

            # use Kernel SHAP to explain test set predictions
            explainer = shap.KernelExplainer(reg.named_steps["reg"], X_new_train, link="logit")
            shap_values = explainer.shap_values(X_new_test, nsamples=100)

            # plot the SHAP values 
            plt.rcParams.update({'font.size': 16})
            f = plt.figure()
            shap.summary_plot(shap_values, X_new_test, link="logit")
            plt.title("Shap values for {}".format(self.modelname))
            plt.show()        


        def plot_permutation_eli5_importance(self,target):
            from eli5.sklearn import PermutationImportance
            import eli5
            #load dataset based on the experiment_number

            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target)))
            if self.studyname == "nn":
                features = reg.named_steps["selectkbest"]
                cols = features.get_support(indices=True)
                X_new = X_test.iloc[:, cols]
                X_test  = X_test .values.astype(np.float32)
                y_test = y_test.values.ravel()
            else: # for non nn models
                # First feature removal step from pipeline
                features = reg.named_steps["cor_removal"]
                cols = features.get_support(indices=True)
                X_new = X_train.iloc[:, cols]
                # Second feature removal step from pipeline
                features = reg.named_steps["rfe_feature_selection"]
                cols = features.get_support(indices=True)
                X_new = X_new.iloc[:, cols]
            perm = PermutationImportance(reg, random_state=1).fit(X_test, y_test)
            data = eli5.show_weights(perm, feature_names = X_new.columns.tolist())
            with open("perm_importance_nn_{}.html".format(target), "w") as file:
                file.write(data.data)

        def plot_permutation_importance(self,target, large_r = False):
            import matplotlib.pyplot as plt

            if self.modelname == "lr":
                    raise ValueError("Your selected model is not a tree model")
            #load dataset based on the experiment_number
 
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target, large_r=False)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target)))

            # First feature removal step from pipeline
            result = permutation_importance(reg, X_train, y_train, n_repeats=5,
                                random_state=42)
            sorted_idx = result.importances_mean.argsort()
            plt.figure()
            plt.style.use('ggplot')
            plt.rcParams.update({'font.size': 16})
            fig, ax = plt.subplots()
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.boxplot(result.importances[sorted_idx].T,
                    vert=False, labels=X_train.columns[sorted_idx])
            ax.set_title("Permutation Importance for {}".format(target), pad=20)
            fig.tight_layout()
            plt.show()

        def plot_test_set_eval(self, target, remove_outliers = False, large_r = False):
            import matplotlib.pyplot as plt
            
            plt.style.use('ggplot')
            fig , ax = plt.subplots()
            plt.rcParams.update({'font.size': 14})
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target, large_r=large_r)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target)))
            if self.studyname == "nn":
                X_test = X_test.values.astype(np.float32)
            pred = reg.predict(X_test)
            if self.studyname == "nn":
                pred = pred.flatten()


            #plot distribution in dataset
            ax2=ax.twinx()
            df_dist = X_train
            df_dist[target] = y_train
            df_x = df_dist[target].value_counts().sort_index(ascending=True)
            ax2.set_ylabel("Samples in training data", labelpad = 10)
            ax2.plot(df_x, linestyle = ":", label = "Samples in training data") 
        
            
            #Create Dataframe to do groupby and mean
            df = pd.DataFrame()
            df["Predictions"] = pred
            df["Measured"] = y_test.values
            df_mean = df.groupby(["Measured"]).mean()
            df_std = df.groupby(["Measured"]).std()
            mean = df_mean["Predictions"].values
            std = df_std["Predictions"].values
            if remove_outliers == True:
                        if target == 'pedestrian':
                            ax.set_xlim(right=40)
                        if target == 'car_or_van_or_suv_or_pickup':
                            ax.set_xlim(right=28)
                        if target == 'motorcycle_or_bicycle':
                            ax.set_xlim(right=8)
                        if target == 'car_or_van_or_suv_or_pickup_moving':
                            ax.set_xlim(right=17)
                        if target == 'truck_or_bus':
                            ax.set_xlim(right=10)
            else:
                ax.set_xlim(right=max(df_mean.index.tolist())+1)
            ax.scatter(x= y_test, y=pred, facecolors='none', edgecolors='gray',label="Prediction")
            ax.errorbar(df_mean.index.tolist(), mean , yerr= std, fmt = ".--k", lw = 1, elinewidth=1, capsize= 3,label="Prediction Mean & Std" )
            #ax[ax_order[idx][0],ax_order[idx][1]].errorbar(df_mean.index.tolist(), mean, [mean - min, max - mean], ".--k", lw = 1, elinewidth=1, capsize= 3,label="prediction mean & std dev")
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '-g', lw=2, label = "Ground Truth")
            ax.set_xlabel('Observed y', labelpad = 10, fontsize = 16)
            ax.set_ylabel('Predicted ŷ', labelpad = 10, fontsize = 16)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            #ax2.legend(lines + lines2, labels + labels2, loc=0)
        
            ax.tick_params(axis='both', labelsize= 13)
            ax2.tick_params(axis='both', labelsize= 13)
            plt.subplots_adjust(hspace= 0.28)
            ax2.legend(lines + lines2, labels + labels2, loc="upper right", prop={"size":12})
            fig.suptitle("Testset predictions for class: {}".format(target),y=0.92)
            plt.tight_layout()
            plt.show()

        def get_training_data_dist(self, target):
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target)
            df_dist = X_train
            df_dist[target] = y_train
            df_x = df_dist[target].value_counts().sort_index(ascending=True)
            df_x.to_csv("training_set_distribution_{}.csv".format(target))

        def get_uncertainty_intervals(self, target, p : Path):
            import matplotlib.pyplot as plt
            plt.style.use('ggplot')
            fig , ax = plt.subplots(2,2)
            ax_order = [[0,0], [0,1], [1,0],[1,1]]
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target)))
            if self.studyname == "nn":
                    X_test = X_test.values.astype(np.float32)
            pred = reg.predict(X_test)
            if self.studyname == "nn":
                pred = pred.flatten()
            #Create Dataframe to do groupby and mean
            df = pd.DataFrame()
            df["Predictions"] = pred
            df["Measured"] = y_test.values 
            df = df.sort_values(by = ['Predictions'])
            df['bins'] = pd.qcut(df['Predictions'], 35, duplicates = 'drop')

            df = df.drop(["Predictions"], axis=1)
            df_mean = df.groupby(["bins"]).mean()
            df_std = df.groupby(["bins"]).std()

            mean = df_mean["Measured"].values
            std = df_std["Measured"].values

            intervals68 = [(round(x - std.tolist()[i], 3), round(x + std.tolist()[i], 3)) for i,x in enumerate(mean.tolist())]
            intervals80 = [(round(x - 1.28 *  std.tolist()[i], 3), round(x + 1.28 * std.tolist()[i], 3)) for i,x in enumerate(mean.tolist())]
            intervals90 = [(round(x - 1.645 * std.tolist()[i], 3) , round(x + 1.645 * std.tolist()[i], 3)) for i,x in enumerate(mean.tolist())]
            mean = [round(x,3) for x in mean]
            bins = df_mean.index.tolist()
            df_complete = pd.DataFrame(data = {
            "bin": bins,
            "mean observed value" : mean,
            "68%  CI" : intervals68, 
            "80%  CI" : intervals80, 
            "90%  CI" : intervals90, 
            })
            df_complete.to_pickle(p.joinpath("{}_{}_experiment_{}.pkl".format(target, self.modelname, self.experiment_number)))

        def load_model(self, target):
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target)))
            return reg

        def get_cv_results(self, target):
            trial = load(os.path.join (self.modeldir, 'optuna', '{}_best_trial_{}.pkl'.format(target, self.studyname)))
            cv_results = trial._user_attrs['crossval scores']
            return cv_results

        def get_best_trial(self, target):
            trial = load(os.path.join (self.modeldir, 'optuna', '{}_best_trial_{}.pkl'.format(target, self.studyname)))
            return trial
            
        def get_study(self, target):
            study = load(os.path.join (self.modeldir, 'optuna', '{}_study_{}.pkl'.format(target, self.studyname)))
            return study

        def get_testset_scores(self, target, large_r = False):
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target, large_r = large_r)
            reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target)))
            if self.studyname == "nn":
                    X_test = X_test.values.astype(np.float32)
            pred = reg.predict(X_test)
            if self.studyname == "nn":
                pred = pred.flatten()
            #Create Dataframe to do groupby and mean
            return r2_score(y_test,pred)
            
        def get_fold_group_information(self):
            from sklearn.model_selection import GroupKFold
            x_train, x_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = 'pedestrian')
            data = x_train
            targets = y_train.values.ravel()
            groups = train_groups  

            group_kfold = GroupKFold(n_splits=5)
            suburb_kfold = group_kfold.split(data, targets, groups)   
            k = 0
            for train, test in suburb_kfold:
                k += 1
                test_g =  groups.values[test]
                train_g = groups.values[train]
                print (f"{k} fold test groups for {np.unique(test_g)} with testsize = {test_g.size}")
                print (f"{k} fold train groups {np.unique(train_g)} with trainsize = {train_g.size}")
            
        def hour_eval(self):
            import matplotlib.pyplot as plt
            from sklearn.metrics import r2_score
            def r2( g ):
                r2 = r2_score( g['Measured'], g['Predicted'] )
                return pd.Series(dict(r2 = r2))
            feature = "hour"
            plt.style.use('ggplot')
            fig , ax = plt.subplots(2,2)
            ax_order = [[0,0], [0,1], [1,0],[1,1]]
            for idx, target in enumerate(self.targets):
                X_train, X_test , y_train , y_test, train_groups, test_groups = train_test_loader(target = target)
                reg = load(os.path.join (self.modeldir, '{}_{}.joblib'.format(self.filename, target)))
                if self.studyname == "nn":
                        X_test = X_test.values.astype(np.float32)
                pred = reg.predict(X_test)
                if self.studyname == "nn":
                    pred = pred.flatten()
                #Create Dataframe to do groupby and mean
                df = pd.DataFrame()
                df["Predicted"] = pred
                df["Measured"] = y_test.values
                df["Feature"] = X_test[feature].values
                df_r2 = df.groupby('Feature').apply(r2)
                ax[ax_order[idx][0],ax_order[idx][1]].set_xlim([-1, 24])
                ax[ax_order[idx][0],ax_order[idx][1]].set_xticks(list(range(24)))
                ax[ax_order[idx][0],ax_order[idx][1]].set_xticklabels(list(range(24)))
                ax[ax_order[idx][0],ax_order[idx][1]].plot(df_r2.index.tolist(), df_r2["r2"], "ok", lw = 1)
                ax[ax_order[idx][0],ax_order[idx][1]].set_ylabel('Coefficient of Determinination')
                ax[ax_order[idx][0],ax_order[idx][1]].set_title("Predictions for {}".format(target.replace("_"," ")))

            plt.subplots_adjust(hspace= 0.28)
            lines_labels = [ax[0,0].get_legend_handles_labels()]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines, labels, loc="upper left")
            fig.suptitle("{} evaluation".format(self.name))
            plt.show()


