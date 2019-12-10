# encoding=utf-8
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



def traitement(filename):

    # load dataset
    pima = pd.read_csv(filename)
    pima.head()
    #split dataset in features and target variable
    feature_cols = ['sexe','age','domaine','statut','fruits et legumes','feculents','laitage','viande','poisson']
    x = pima[feature_cols] # Features
    y = pima.style_vestimentaire # Target variable

    # return split dataset into training set and test set

    return train_test_split(x, y, test_size=0.2, random_state=0) # 80% training and 20% test
    
def tree_classifier(x_train, x_test, y_train, y_test):
	# Create Decision Tree classifer object
    #clf = DecisionTreeClassifier()

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

    # Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(x_test)
    
    print("***************************************** ")
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    print("mse: ",mean_squared_error(y_test, y_pred))
    print("mae: ",mean_absolute_error(y_test, y_pred))
    graphic(clf)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

def cross_validation_classifier(x_train, x_test, y_train, y_test):

    pgrid = {"max_depth": [1, 2, 3, 4, 5, 6, 7],
      "min_samples_split": [2, 3, 5, 10, 15, 20]}

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=pgrid, scoring='neg_mean_squared_error', cv=10)
    grid_search.fit(x_train, y_train)
    
    y_pred = grid_search.best_estimator_.predict(x_test)
    #grid_search.best_estimator_.score(x_test, y_test)
    print("***************************************** ")
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("mse: ",mean_squared_error(y_test, y_pred))
    print("mae: ",mean_absolute_error(y_test, y_pred))
    print("best_param:", grid_search.best_params_)


#from sklearn.tree import export_graphviz 
from IPython.display import Image  
from sklearn import tree
import pydotplus


def graphic(clf):

    """
    dot_data = tree.export_graphviz(clf, out_file=None,
            feature_names=feature_cols,
            class_names=['0','1'],
            filled=True, rounded=True,
            special_characters=True)
    """	
    with open("iris.dot", 'w') as f:
    	f = tree.export_graphviz(clf, out_file=f)
#Ne marche pas pour le moment
    feature_cols = ['sexe','age','domaine','statut','fruits et legumes','feculents','laitage','viande','poisson']
    dot_data = tree.export_graphviz(clf, out_file=None,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=None, label='style_vestimentaire')
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('style_vestimentaire.png')
    Image(graph.create_png())

#if __name__ == '__main__':
