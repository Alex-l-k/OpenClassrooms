import numpy as np
import pandas as pd
import joblib

pipeline = joblib.load('modele_billets.pkl')

features = ['diagonal', 
            'height_left', 
            'height_right', 
            'margin_low', 
            'margin_up', 
            'length']

def predire_billet(diagonal, height_left, height_right, margin_low, margin_up, length):

    billet = pd.DataFrame({
    'diagonal': [diagonal],
    'height_left': [height_left],
    'height_right': [height_right],
    'margin_low': [margin_low],
    'margin_up': [margin_up],
    'length': [length]
    })
    
    prediction = pipeline.predict(billet)[0]

    if prediction == 1:
        return "Vrai billet"
    else:
        return "Faux billet"
    



def predire_fichier(chemin_csv):
    
    df = pd.read_csv(chemin_csv, sep=';')

    predictions = pipeline.predict(df[features])

    df['prediction'] = ['Vrai billet' if p == 1 else 'Faux billet' for p in predictions]

    return df[['prediction']]


    
print(predire_billet(diagonal=171.50,
    height_left=104.20,
    height_right=104.10,
    margin_low=5.50,
    margin_up=3.40,
    length=111.50))


# print(predire_fichier('billets_production.csv'))