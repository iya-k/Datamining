#!/usr/bin/env python3
# encoding=utf-8

import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from IPython.display import Image 
from sklearn.externals.six import StringIO 
from sklearn import tree
import pydotplus

def processing(in_put,output):
#def split_lines(entree,sortie):

    file_in = open(in_put,'r')
    entree = csv.reader(file_in , delimiter = ',')
    file_out = open(output, 'w')
    sortie = csv.writer(file_out , delimiter=',')
    rows = []
    #recuperer les colonnes excepter la premiere
    for row in entree:
        rows.append(row[1:])
    #traitement des données
    for line in rows:
        #print(i)
        for i in range(len(line)):

            if(line[i] == 'Homme'):
                line[i] = 111
            elif(line[i] == 'Femme'):
                line[i] = 122
            if(line[i] == '18-25 ans'):
                line[i] = 18
            elif(line[i] == '26-35 ans'):
                line[i] = 26
            elif(line[i] == '36-45 ans'):
                line[i] = 36
            elif(line[i] == '+ 45 ans'):
                line[i] = 45
            if(line[i] == 'Informatique & Télécommunication'):
                line[i] = 331
            elif(line[i] == 'Comptabilité & Marketing'):
                line[i] = 332
            elif(line[i] == 'Audio Visuel'):
                line[i] = 333
            elif(line[i] == 'Industrie'):
                line[i] = 334
            elif(line[i] == 'Bâtiment'):
                line[i] = 335
            elif(line[i] == 'Medical'):
                line[i] = 336
            elif(line[i] == 'Banque & Assurance'):
                line[i] = 337
            elif(line[i] == 'Transport'):
                line[i] = 338
            elif(line[i] == 'Mode'):
                line[i] = 339
            if(line[i] == 'Etudiant(e)'):
                line[i] = 441
            elif(line[i] == 'Cadre'):
                line[i] = 442
            elif(line[i] == 'ouvrier(ère)'):
                line[i] = 443
            elif(line[i] == 'Indépendant(e)'):
                line[i] = 444
            if(line[i] == 'Tous les jours'):
                line[i] = 511
            elif(line[i] == 'Une fois par semaine'):
                line[i] = 522
            elif(line[i] == 'Une fois par mois'):
                line[i] = 533
            elif(line[i] == 'Jamais'):
                line[i] = 544
            
        sortie.writerow(line)
    file_out.close()
    file_in.close()

def style_Vestimentaire(row, style):

    if(style == 'naturel'):
        if(row[0] == "Naturel/sportive"):
            row[0]= "0"
        else:
            row[0] = "1"

    if(style == 'classique'):
        if(row[0] == "Classique"):
            row[0]= "0"
        else:
            row[0] = "1"
            
    if(style == 'minimaliste'):
        if(row[0] == "Minimaliste"):
            row[0]= "0"
        else:
            row[0] = "1"

    if(style == 'romantique'):
        if(row[0] == "Romantique"):
            row[0]= "0"
        else:
            row[0] = "1"

    if(style == 'dramatique'):
        if(row[0] == "Dramatique/Créative"):
            row[0]= "0"
        else:
            row[0] = "1"

    if(style == 'tendance'):
        if(row[0] == "Tendance/fashionista"):
            row[0]= "0"
        else:
            row[0] = "1"

    if(style == 'neutre'):
        if(row[0] == "Naturel/sportive"):
            row[0]= "1"
        elif(row[0] == 'Classique'):
            row[0]= "2"
        elif(row[0] == 'Minimaliste'):
            row[0]= "3"
        elif(row[0] == 'Romantique'):
            row[0]= "4"
        elif(row[0] == "Dramatique/Créative"):
            row[0]= "5"
        elif(row[0] == "Tendance/fashionista"):
            row[0]= "6"

    return row

def transformation(in_put, output_style, style):#, output2, output3, output4, output5, output6):
    
    file_in = open(in_put,'r')
    entree = csv.reader(file_in , delimiter = ',')
    file_out = open(output_style, 'w')
    sortie = csv.writer(file_out , delimiter=',')

    rows = []
    cpt = 0
    for row in entree:
        if(cpt == 0):
            rows.append(row)
            cpt += 1
        else:
            rows.append(style_Vestimentaire(row, style))

    for line in rows:
        sortie.writerow(line)

    file_out.close()
    file_in.close()

#partager les donnees en train et test
def split_data(filename):
    
    print("************",filename, "******************* ")
    cols_names = ['style_vestimentaire','sexe','age','domaine','statut']#,'fruits et legumes','feculents','laitage','viande','poisson']
    # load dataset
    dataset = pd.read_csv(filename, sep=',', encoding='utf-8', usecols = cols_names)
    #print(dataset.head())
    feature_cols = ['sexe','age','domaine','statut']#,'fruits et legumes','feculents','laitage','viande','poisson']
    data = dataset[feature_cols] # Features
    label = dataset.style_vestimentaire # Target variable
    # return split dataset into training set and test set; 80% training and 20% test
    return data, label

from matplotlib.pyplot import matshow, colorbar, title, show

def tree_regressor(x_train, x_test, y_train, y_test):
	# Create Decision Tree classifer object
    model = DecisionTreeRegressor(criterion="mse", max_depth=4, min_samples_split=2, random_state=0)
    # Train Decision Tree Classifer
    model = model.fit(x_train,y_train)
    #Predict the response for test dataset
    y_pred = model.predict(x_test)
    print("*************** tree_classifier ***************** ")
    print("Mean squared error: ",round(mean_squared_error(y_test, y_pred),2))
    print("Mean absolute error: ",round(mean_absolute_error(y_test, y_pred),2))
    return model

def random_forest(x_train, x_test, y_train, y_test):
    print("********  random_forest *******")
    regressor = RandomForestRegressor(n_estimators=6, random_state=0, criterion='mse') 
    params = {
    'n_estimators': [5, 6],
     'min_samples_split': [4, 5],
     'min_samples_leaf': [3, 4, 5],
      'max_features': ['sqrt'],
      'max_depth': [8, 9],
       'bootstrap': [True]}
    
    grid_search = GridSearchCV(estimator = regressor, param_grid=params, cv=3)
    grid_search.fit(x_train, y_train)
    y_pred = grid_search.best_estimator_.predict(x_test)
    
    print("MSE: ", round(mean_squared_error(y_test, y_pred), 3))
    print("MAE: ", round(mean_absolute_error(y_test, y_pred), 3))
    print("score: ", round(grid_search.score(x_test, y_test), 3))

    print("best_param:", grid_search.best_params_)
    return grid_search

def graphic(model, image):
    dot_data = export_graphviz(model, out_file=None,  
                filled=True, rounded=True, precision = 1,
                special_characters=True,feature_names = ['style_vestimentaire'])
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_png(image)
    Image(graph.create_png())

if __name__ == '__main__':
    
    style_image = ['./images/style_naturel.png', './images/style_romantique.png',
     './images/style_dramatique.png', './images/style_tendance.png',
      './images/style_classique.png', './images/style_minimaliste.png']
    fichier = ['./donnees/dataset_naturel.csv', './donnees/dataset_romantique.csv',
     './donnees/dataset_dramatique.csv', './donnees/dataset_tendance.csv',
      './donnees/dataset_classique.csv', './donnees/dataset_minimaliste.csv']
    style = ['naturel', 'romantique', 'dramatique','tendance', 'classique', 'minimaliste']
    i = 0
    processing('./donnees/dataset.csv','./donnees/dataset_global.csv')
    transformation('./donnees/dataset_global.csv', './donnees/dataset_final.csv', 'neutre')

    for file in fichier:

        transformation('./donnees/dataset_global.csv', file, style[i])
        data, label = split_data(file)
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)
        model_ = tree_regressor(x_train, x_test, y_train, y_test)
        graphic(model_, style_image[i])
        print("score: ", model_.score(x_test, y_test))
        i += 1
    


    data, label = split_data('./donnees/dataset_final.csv')
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)
    model_= random_forest(x_train, x_test, y_train, y_test)
    model_ = tree_regressor(x_train, x_test, y_train, y_test)
    graphic(model_, './images/style_vestimentaire.png')

    
