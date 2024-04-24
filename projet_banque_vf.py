import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px 

df = pd.read_csv('bank.csv')

st.title("Améliorer l'efficacité d'une campagne de télémarketing d'une banque")

pages=["Présentation du projet", "Exploration du jeu de données", "Visualisation des données", "Nos partis pris", "Modélisation", "Interprétation", "Clusters"]
page=st.sidebar.radio("Aller vers", pages)

multi = '''
Avril 2024
DataScientest - Data Analyse
Promotion Février 2024 - Bootcamp
'''

st.sidebar.markdown(multi)

st.sidebar.write("## L'équipe de recherche")

multi2 = '''
Alice Rayon     
Robin Guiavarch    
Mathias Vivier    
Clément Fouchier   
'''

st.sidebar.markdown(multi2)

if page == pages[0] : 
  st.write("### Contexte")

  st.write("Nous avons à disposition des données personnelles sur des clients d’une banque qui ont été “télémarketés” pour souscrire un produit que l’on appelle un 'dépôt à terme'. \
         Lorsqu’un client souscrit ce produit, il place une quantité d’argent dans un compte spécifique et ne pourra pas toucher ces fonds avant l’expiration du terme. \
         En échange, le client reçoit des intérêts attractifs de la part de la banque à la fin du terme.")

  st.write("À partir des informations sur les personnes contactées, nous devons trouver les caractéristiques partagées par les personnes les plus enclines à souscrire au dépôt à termes. \
         Une fois ce travail effectué, il sera possible grâce au Machine Learning de prédire la probabilité qu’un client donné souscrive un dépôt à terme.\n \
         Ce travail a pour but d’optimiser la campagne de Marketing Direct de la banque, en améliorant l'efficacité de son ciblage, pour appeler en priorité les clients plus susceptibles de souscrire au dépôt. \
         Avec, in fine, des gains commerciaux et financiers.")
  
  st.write("### Objectifs")

  st.write("Tout au long de ce travail de recherche, nous poursuivrons deux objectifs :")

  multi = '''
  - Objectif 1 : Délivrer des prédictions les plus fiables possibles
  - Objectif 2 : Comprendre les ressorts qui amènent les individus à souscrire un dépôt à terme

  '''

  st.markdown(multi)


  st.write("### Plan d'investigation")
  
  multi = '''
  1. Exploration du jeu de données : compréhension du jeu de données, gestion des valeurs manquantes
  2. Visualisation des données : approfondissement au travers de graphiques, relations entre les données et la variable cible
  3. Partis pris : nos partis pris pour la réalisation du modèle de Machine Learning
  4. Modélisation : modèles de Machine Learning et performances associées
  5. Interprétation : compréhension des performances
  6. Pour aller plus loin : constitution de clusters pour améliorer l'efficacité globale des campagnes

  '''
  st.markdown(multi)

if page == pages[1]:
  st.write("### 1. Exploration du jeu de données \n")

  st.write("Notre jeu de données est constitué de différents types de données : socio-démographiques, bancaires, liées aux campagnes précédentes, liées à la campagne actuelle.")

  st.write("Apperçu du jeu de données :")
  st.dataframe(df.head(10))

  st.write("Nombre de lignes :", df.shape[0])

  st.write("Nombre de colonnes :", df.shape[1], "\n")

  if st.checkbox("Afficher les nombre de valeurs manquantes par colonne"):
    st.dataframe(df.isna().sum())

  df = df.replace("unknown", np.nan)

  if st.checkbox("Afficher les nombre de valeurs manquantes par colonne - sans 'unknown', et en pourcentages"):
    st.dataframe((df.isna().sum())/df.shape[0]*100)

  month_values_contact = (df.loc[df["contact"].isna()]["month"].value_counts(normalize = 1))*100
  month_values = (df["month"].value_counts(normalize = 1))*100

  data = {
    "Valeurs manquantes" : month_values_contact, 
    "Ensemble" : month_values,
    "Différence" : month_values_contact - month_values
  }

  if st.checkbox("Exporation des valeurs manquantes de la variable contact"):
     st.write("Regardons le pourcentage des valeurs manquantes chez les individus selon le mois du dernier appel enregistré, ainsi que pour l'ensemble des données :")
     st.dataframe(pd.DataFrame(data).sort_values(by = "Valeurs manquantes", ascending = False))
     st.write("La majeure partie des valeurs manquantes provient du mois de mai (60%), liée à une surcharge d'appels (25% du total des appels). Il semble que les télémarketeurs aient perdu l'habitude de demander le mode de contact privilégié. \n")

     st.write("Vérifions l'impact sur la variable cible :")

     st.image("contact.png")

     st.write("Nous voyons un impact : nous décidons de la conserver pour le modèle prédictif, au moins dans un premier temps.")

     
  if st.checkbox("Exporation des valeurs manquantes de la variable poutcome (résultats des précédentes campagnes)"):
     st.write("Après investigation, nous remarquons que les 75% de valeurs manquantes sont les individus non appelés pour des campagnes précédentes.")

     st.write("Vérifions l'impact sur la variable cible :")

     fig = px.histogram(df, x="poutcome", color="deposit", barnorm='percent', text_auto='.2f', title="Nombre de conversions selon les résultats aux précédentes campagnes")
     st.plotly_chart(fig, use_container_width=True, theme = None)

     st.write("Un taux de souscription très fort chez les individus qui avaient accepté la souscription lors de précédentes campagnes. Nous choisissons de conserver, pour le modèle prédictif, les individus qui n'ont pas été appelés pour des campagnes précédentes, ou qui n'avaient pas déjà dit oui.")

     st.write("Par conséquent : nous supprimons les individus étiquetés 'success' à cette variable du jeu de données, et supprimons cette colonne car il y a trop de valeurs manquantes.")

     df_new = df.loc[df.poutcome != "success"]

     st.write("Nouveau nombre de lignes :", df_new.shape[0])

df = df.replace("unknown", np.nan)
df_new = df.loc[df.poutcome != "success"]

if page == pages[2] :
  st.write("### 2. Visualisation des données")

  st.write("Le Machine Learning est plus performant lorsque la variable cible est équilibrée. Vérifions :")

  fig = px.histogram(df_new, x="deposit", title="Pourcentage de souscriptions d'un dépôt à terme", histnorm = "probability density")
  st.plotly_chart(fig, use_container_width=True)

  st.write("Regardons maintenant plus en détail la relation de certaines variables clé avec la variable cible :")

  if st.checkbox("Variable age"):
    fig = px.box(data_frame = df_new, x="age", title = "Distribution de l'âge", color = "deposit")
    st.plotly_chart(fig, use_container_width=True, theme = None)
        
    st.write("Ce graphique semble indiquer une distribution plus importante aux extremités chez les individus qui ont souscrit. Vérifions :")

    st.image("age.png")
    st.write("*Aide à la lecture : les barres grises indiquent le chevauchement des deux catégories. Le dessus des barres est bleu quand le décompte des individus qui ont un deposit dépasse celui des individus qui n’en possèdent pas. Et inversement.*")

    st.write("Ce nouveau graphique confirme la distribution plus forte aux extrémités des individus qui ont un deposit.")

  if st.checkbox("Variable education"):
    fig = px.histogram(df_new, x="education", color="deposit", barnorm='percent', text_auto='.2f', category_orders = {"education" : ["primary", "secondary", "tertiary"]}, title="Deposit selon le niveau d'éducation")    
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("Plus le niveau d'éducation est fort, plus les chances de souscription sont importantes.")

  if st.checkbox("Variable job"):
    fig = px.histogram(df_new, x="job", color="deposit", barnorm='percent', text_auto='.2f', title="Deposit selon le job")    
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("Des taux de souscription plus importants chez les inactifs : étudiants, retraités, sans-emplois ; et dans une moindre mesure chez les cadres.")

  if st.checkbox("Variable martial (statut marital)"):
    fig = px.histogram(df_new, x="marital", color="deposit", barnorm='percent', text_auto='.2f', title="Deposit selon le statut marital")    
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("Les célibataires sont plus susceptibles de souscrire que les individus mariés.")

  if st.checkbox("Variable balance (montant annuel sur les comptes en moyenne)"):
    st.write("Pour ce graphique, nous conseillons de zoomer sur le graphique en sélectionnant la zone souhaitée.")
    fig = px.box(data_frame = df_new, x="balance", title = "Distribution de l'équilibre des comptes sur l'année", color = "deposit")
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("La balance des individus qui ont souscrit est supérieure à ceux qui n'ont pas souscrit, avec une médiane de 712€ vs 410€.")

  if st.checkbox("Variable housing (crédit immobilier en cours)"):
    fig = px.histogram(df_new, x="housing", color="deposit", barnorm='percent', text_auto='.2f', category_orders = {"housing" : ["no", "yes"]}, title="Deposit selon la possession ou non d'un crédit immobilier")    
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("Une influence importante sur la souscription d'un dépôt à terme : avoir un crédit immobilier réduit considérablement les chances de souscrire.")

  if st.checkbox("Variable loan (crédit à la consommation en cours)"):
    fig = px.histogram(df_new, x="loan", color="deposit", barnorm='percent', text_auto='.2f', category_orders = {"loan" : ["no", "yes"]}, title="Deposit selon la possession ou non d'un crédit à la consommation")    
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("Une influence sur la souscription, mais moindre que le crédit immobilier.")

  if st.checkbox("Variable month (mois du dernier appel)"):
    fig = px.histogram(df_new, x="month", color="deposit", category_orders = {"month" : ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]}, title="Saisonalité du mois de dernier contact")
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("Nous remarquons un nombre important de derniers contacts en mai, juin, juillet et août. Certains mois semblent dédiés à la relance d'individus qui avaient l'air intéressés comme mars, septembre, octobre et décembre.")

  if st.checkbox("Variable weekday (dernier jour d'appel selon le jour de la semaine)"):

    st.write("A partir de la variable 'day', qui répertorie le dernier jour de l'appel sur les 28 à 31 jours d'un mois, nous avons créé la variable weekeday, qui identifie les jours de la semaine. Voici son influence :")

    st.image("weekday.png")
    
    multi = '''
    **1er constat** :  la majorité des appels sont passés entre lundi et jeudi (environ 2000 par jour). 

    **2ème constat** : plus la semaine avance, plus le taux de conversion décroît (de 49% le Lundi à 29% le Vendredi). Les Jeudis et Vendredis semblent être des jours à éviter.
    '''
    st.markdown(multi)

  if st.checkbox("Variable duration (durée du dernier appel, en secondes)"):
    fig = px.box(data_frame = df_new, x="duration", title = "Distribution de la durée du dernier appel", color = "deposit")
    st.plotly_chart(fig, use_container_width=True, theme = None)
    st.write("Une relation très forte entre la variable cible et la durée du dernier appel. Logique, puisque plus l'appel prend du temps, plus il y a de chances qu'il s'agisse d'une souscription. Il s'agit en quelques sortes d'une conséquence de la variable cible. Nous supprimons la variable.")
    st.write("De plus, il s'agit de données que nous n'avons pas avant le début de la campagne, donc qui ne peuvent pas être utilisées pour prédire les soucripteurs de prochaines campagnes.")
  
  st.write("\nToutes les variables présentées ont une influence statistiquement validée sur la variable cible.")
  st.write("\nPar ailleurs, nous constatons un nombre important de valeurs extrêmes (caractérisées par les points sur les graphiques liés aux variables quantitatives), sans qu'elles soient pour autant aberrantes. Nous avons veillé à limiter leur impact sur le modèle en utilisant la méthode de normalisation adaptée.")

if page == pages[3] :
  st.write("### 3. Nos partis pris pour la mise en place du Machine Learning \n")

  st.write("- Un modèle *pertinent*, en conservant les données les plus intéressantes pour le modèle prédictif, et ce même si elles réduisent la performance finale.")

  st.write("- Un modèle *utilisable* pour de futures campagnes : en ne conservant que des données que nous avons à disposition avant la lancement d'une campagne.")

  st.write("- Un modèle *généralisable* : en limitant le surapprentissage et en appliquant des méthodes de validation croisée.")

  st.write("- Un modèle *performant*, quitte à le rendre moins interprétable...")

  st.write("... mais sans pour autant abandonner l'ambition d'en comprendre les ressorts, et de connaître le profil des individus les plus suceptibles de souscrire.")

if page == pages[4] : 
    st.write("## 4. Modélisation")
    st.write("")    
    if st.checkbox("*Jeu de données initial*") :
        st.dataframe(df.head(10)) 
     
        # 1) Présentation des indicateurs de performances du modèle
    st.write("")
    st.write("#### Présentation des indicateurs de performance d'un modèle Machine Learning")

    # Score de rappel de la classe 1 - la sensibilité

    sensibilite = """
    Le score de rappel de la classe 1 ou la sensibilité est une métrique qui permet de savoir le pourcentage de soucriptions bien prédites par notre modèle.\n
    Parce que nous ne voulons surtout pas passer à côté de clients potentiels qui auraient pu souscrire au deposit, c'est l'une des métriques que nous chercherons à optimiser.\n
    """
    if st.checkbox("Score de rappel de la classe 1 : la sensibilité") :
        st.write(sensibilite)


    # Métrique fait maison - metric_balance

    def metric_balance():
        
        metric_bal1 = """
        Metric_balance est une métrique qui pénalisera les modèles qui n'ont pas réussis à bien prédire les clients dont le solde de compte est le plus élevé. 
        En effet, nous voulons à tout prix éviter de passer à côté de clients susceptibles non seulement de souscrire, mais en plus pour des sommes importantes. 
        Le solde de compte est représenté par la variable balance dans notre jeu de données.\n
        Voici comment elle se construit:
        """
        st.write(metric_bal1)

        st.image("metric_balance.png", use_column_width=True)

        metric_bal2 = """
        Le schéma représente la distribution de la variable balance que l’on sépare selon ses déciles.
        Si jamais le modèle prédit un faux négatif pour un client qui a un solde au dessus du décile 5 (à savoir la médiane), 
        la valeur de la pénalité se déplace de 0,1 jusqu’à 1,5. Par contre, en dessous de la médiane,
        on ajoute des pénalités négatives (donc des bénéfices) de -0,1 à -1,5.\n
        Comment interpréter cette métrique?\n
        Metric_balance = 1 - penalty\n
        - Si metric_balance = 1 : alors le modèle n’a aucune influence. Il ne va pas mieux prédire les clients qui ont un solde élevé. \n
        - Plus metric_balance est grande face à 1, plus le modèle limite le nombre de faux négatifs pour les clients ayant un solde élevé.\n
        - Inversement lorsque metric_balance décroît.
        """
        st.write(metric_bal2)

    if st.checkbox("Métrique fait maison : la metric balance pour maximiser les individu qui ont le compte en banque le plus rempli") :
        metric_balance() 

    # Courbe sensibilité en fonction du seuil de classification - indicateurs p80 et r40
        
    def p80_r40():
        
        # Courbe sensibilité en fonction du seuil de classification
        
        courbe_sens = """
        Il y a 2 façons de présenter les prédictions d'un modèle de machine learning. Soit de manière binaire, dans notre cas 1 pour les souscripteurs,
        et 0 pour les non souscripteurs. Soit en termes de probabilité d'appartenance à la classe 1 (souscription).\n
        Le seuil de classification est le pallier statistique qui permet au modèle de classer les individus selon leur classe.
        Par défaut, le seuil est à 50%, mais il peut être modifié pour intégrer plus ou moins d’individus dans la classe 1 (soit les souscripteurs). 
        En variant ce seuil de classification, on varie de fait le score de rappel de la classe 1, aussi appelé la sensibilité."""
        st.write(courbe_sens)
        
        st.write("##### La courbe")
        st.write('Pour chaque modèle, nous pouvons donc tracer une courbe représentant la sensibilité en fonction du seuil de classification. En voici un exemple pour un modèle Random Forest avec tous ses hyperparamètres par défaut:')

        st.image("courbe_sens_class.png", width = 600)

        st.write("###### Pourquoi cette courbe témoigne-t-elle de la performance du modèle ?")
        
        exp_p80_r40 = """
        L'idée reste toujours de gonfler ce score de rappel de la classe 1, c'est à dire de prédire le plus grand nombre possible de souscripteurs.
        Pour augmenter le score de rappel, nous pouvons abaisser le seuil de classification, car plus le nombre d'individus classés 
        comme souscripteurs augmente, plus le score de rappel va donc croître. Et c'est ce que montre cette courbe.\n
        Un modèle performant sera donc un modèle qui nous permet d'atteindre un score de rappel élevé en n'ayant pas besoin de baisser de beaucoup le 
        seuil de classification.
        """

        st.write(exp_p80_r40)


    if st.checkbox("Faire varier le seuil de classification des individus susceptibles de souscrire") :
        p80_r40()


    # Data preprocessing

    st.write("")        
    st.write("#### Réduction du jeu de données : choix des variables")

    df = pd.read_csv("bank.csv")
    
    def partie_2():

        reduction = """
        En plus de la réduction du Dataset détaillée dans les pages précédantes, nous supprimerons aussi les variables 'month' et 'campaign', car elles concernent la campagne en cours.
        En effet, lors de l'implémentation de notre machine learning, nous n'aurons pas accès à ces données.
        """
        st.write(reduction)

        st.write("##### Réduction supplémentaire pour minimiser le surapprentissage")

        dim_data = """
        La réduction du Dataset - en enlevant les colonnes / variables qui ont le moins d'influence sur le modèle - est un bon moyen de diminuer
        le surapprentissage du modèle.\n
        Nous créons plusieurs dimensions de données pour tester l'influence sur les résultats et le surapprentissage. Il y en a 6 au total, de 0 (défaut, toutes les variables) à 5 (uniquement 4 variables).
        Vous pourrez comparer par vous-même sur la partie suivante.
        """
        st.write(dim_data)

    if st.checkbox("Cliquez pour en savoir plus sur notre choix des variables retenues") :
        partie_2()

    # Code du preprocessing du jeu de donnée
######################################################################################################

    #importationrtaion du dataframe
    df_init = pd.read_csv("bank.csv", sep=",", header=0)
    #    print(df_init.head())

    #si besoin : reset le df aux données initiales
    df=df_init
    #    print(df_init.info())

    #I.cleaning avant split

    #Date
    df = df.drop(columns=['month', 'day'])

        #vérification des étapes précédentes
    #    print(df.info())
    #.   print(df.shape)

    #Colonne poutcome

        #Supression des "yes" de la colonne poutcome
    df = df.loc[df['poutcome'] != 'success']

        #vérification des étapes précédentes
    #print(df['poutcome'].value_counts())

        #Suppression de la colonne
    df = df.drop(columns=['poutcome'])

    #Colonne duration
        #Suppression de la colonne
    df = df.drop(columns=['duration'])
        #vérification des étapes précédentes
    #print(df.columns)


    # remplacement des yes / no
        #remplacement
    df = df.replace(to_replace=["yes","no"], value=[1,0])
        #vérification des étapes précédentes
    #print(df.info())
    #for columns in df.columns:
        #print(df[columns].value_counts())

        
    #Colonne contact
        #Suppression de la colonne    
    df = df.drop(columns=['contact'])

    #Colonne campaign

    df = df.drop(columns=['campaign'])


    #II. split
    X = df.drop(columns="deposit")
    y = df['deposit']
    from sklearn.model_selection import train_test_split # type: ignore

    # Taille du jeu de test fixé à 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=42)

    #III. Remplacement valeurs manquantes et Numérisation

    from sklearn.impute import SimpleImputer # type: ignore

    imputer = SimpleImputer(missing_values="unknown", strategy='most_frequent')


    # Variables 'job' et 'education' - valeurs manquantes
    cols = ['job','education']
    X_train[cols] = imputer.fit_transform(X_train[cols])
    X_test[cols] = imputer.transform(X_test[cols])

    #encodage ordinal variable education
    from sklearn.preprocessing import OrdinalEncoder # type: ignore

    categories = [["primary", "secondary", "tertiary"]]
    oe = OrdinalEncoder(categories=categories)

    cols3 = ['education']
    X_train[cols3] = oe.fit_transform(X_train[cols3])
    X_test[cols3] = oe.transform(X_test[cols3])

    #Encodage onehotencoder variables job et marital
    col_ohe = ['job','marital']

    X_train_ohe = pd.get_dummies(X_train[col_ohe], drop_first = True)
    X_test_ohe = pd.get_dummies(X_test[col_ohe], drop_first = True)

    X_train = X_train.join(X_train_ohe)
    X_train = X_train.drop(col_ohe, axis = 1)

    X_test = X_test.join(X_test_ohe)
    X_test = X_test.drop(col_ohe, axis = 1)


    #Standardisation

    from sklearn.preprocessing import StandardScaler # type: ignore

    col_num = ["age", "balance", "pdays",'previous']

    scaler = StandardScaler()

    X_train[col_num] = scaler.fit_transform(X_train[col_num])

    X_test[col_num] = scaler.transform(X_test[col_num])



######################################################################################################
######################################################################################################

# Partie 3) Environnement d'essai - 1ère approche

######################################################################################################
######################################################################################################

    st.write("")
    st.write("#### A vous de jouer : construisez votre propre modèle !")
    st.set_option('deprecation.showPyplotGlobalUse', False)

# Définition metric_balance
######################################################################################################

    import statistics

    deciles = statistics.quantiles(X_train['balance'], n=10)


    def compute_score(y_pred):
        

        somme = 0
        for i in (y_pred<y_test).index:
            
            if (y_pred<y_test).loc[i]:
                
                if (X_test.loc[i,'balance']<=deciles[0]):
                    somme = somme - 1.5 
                if (X_test.loc[i,'balance']>deciles[0]) and (X_test.loc[i,'balance']<=deciles[1]):
                    somme = somme - 1.0 
                if (X_test.loc[i,'balance']>deciles[1]) and (X_test.loc[i,'balance']<=deciles[2]):
                    somme = somme - 0.6 
                if (X_test.loc[i,'balance']>deciles[2]) and (X_test.loc[i,'balance']<=deciles[3]):
                    somme = somme - 0.3 
                if (X_test.loc[i,'balance']>deciles[3]) and (X_test.loc[i,'balance']<=deciles[4]):
                    somme = somme - 0.1   
                if (X_test.loc[i,'balance']>deciles[4]) and (X_test.loc[i,'balance']<=deciles[5]):
                    somme = somme + 0.1 
                if (X_test.loc[i,'balance']>deciles[5]) and (X_test.loc[i,'balance']<=deciles[6]):
                    somme = somme + 0.3 
                if (X_test.loc[i,'balance']>deciles[6]) and (X_test.loc[i,'balance']<=deciles[7]):
                    somme = somme + 0.6 
                if (X_test.loc[i,'balance']>deciles[7]) and (X_test.loc[i,'balance']<=deciles[8]):
                    somme = somme + 1.0 
                if (X_test.loc[i,'balance']>deciles[8]):
                    somme = somme + 1.5 
        
        return (1 - somme/len(X_test))

# Metriques accuracy, overfitting, rappel classe 1 + matrice de confusion + classification report
######################################################################################################

# Accuracy et classification report
    from sklearn.metrics import classification_report # type: ignore
    from sklearn.metrics import accuracy_score

    # overfitting
    def overfit_score(y_pred_train, y_pred_test):
        overfit = accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test)
        return overfit

    # rappel classe 1 
    from sklearn.metrics import recall_score

    def recall1_score(y_pred_test):
        return recall_score(y_test, y_pred_test, pos_label=1)

    # matrice de confusion
    def conf_matrix(y_pred_test):
        matrix = pd.crosstab(y_pred_test, y_test, colnames =  ['Real classes'], rownames = ['Predicted classes'])
        return matrix


# courbe sensibilité en fonction du score de rappel
######################################################################################################

    # roc_curve
    from sklearn.metrics import roc_curve, auc

    def courbe_sensibilite(probs_test):
    

        # DataFrame df_seuil_recall1 
        fpr, recall_1, seuils = roc_curve(y_test, probs_test[:,1], pos_label=1)        
        df_seuil_recall1 = pd.DataFrame({'Recall_1': recall_1, 'Seuils_proba' : seuils})

        # Traçage de la courbe

        # On utilise le dataframe qui regroupe les scores de rappel 1 et seuils de classification
        y_serie = df_seuil_recall1['Recall_1']
        x_serie = df_seuil_recall1['Seuils_proba']

        x_list = list(np.arange(0.3,0.8,0.1))
        y_list = [df_seuil_recall1.loc[df_seuil_recall1['Seuils_proba'] <= i].head(1)['Recall_1'].iloc[0] for i in x_list]

        # On trace le barplot ainsi que la courbe
        fig = plt.figure(figsize = (8,4))    
        plt.plot(x_serie, y_serie, color = 'red')
        plt.bar(x_list,y_list, color = 'blue', width = 0.07, alpha = 0.3)
        for i in [0.3,0.4,0.5,0.6,0.7]:
            plt.text(i, y_list[int(i*10)-3] + 0.01, round(y_list[int(i*10)-3],2))

        plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        plt.xlim(0.0,1)
        plt.ylim(0,1.1)
        plt.title("Score rappel classe 1 en fonction du seuil de classification")
        plt.xlabel("Seuil de classification")
        plt.ylabel("Score de rappel classe 1")

        st.pyplot(fig)

    def courbe_sensibilite_v2(probs_test):
    

        # DataFrame df_seuil_recall1 
        fpr, recall_1, seuils = roc_curve(y_test, probs_test[:,1], pos_label=1)        
        df_seuil_recall1 = pd.DataFrame({'Recall_1': recall_1, 'Seuils_proba' : seuils})

        # Traçage de la courbe

        # On utilise le dataframe qui regroupe les scores de rappel 1 et seuils de classification
        y_serie = df_seuil_recall1['Recall_1']
        x_serie = df_seuil_recall1['Seuils_proba']

        x_list = list(np.arange(0.4,0.7,0.1))
        y_list = [df_seuil_recall1.loc[df_seuil_recall1['Seuils_proba'] <= i].head(1)['Recall_1'].iloc[0] for i in x_list]

        # On trace le barplot ainsi que la courbe
        fig = plt.figure(figsize = (8,4))    
        plt.plot(x_serie, y_serie, color = 'red')
        plt.bar(x_list,y_list, color = 'blue', width = 0.07, alpha = 0.3)
        for i in [0.4,0.5,0.6]:
            plt.text(i, y_list[int(i*10)-4] + 0.01, round(y_list[int(i*10)-4],2))

        plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        plt.xlim(0.0,1)
        plt.ylim(0,1.1)
        plt.title("Score rappel classe 1 en fonction du seuil de classification")
        plt.xlabel("Seuil de classification")
        plt.ylabel("Score de rappel classe 1")

        st.pyplot(fig)




# Indicateur p80 et r40
######################################################################################################

    def p80_score(probs_test):
        
        # DataFrame df_seuil_recall1 
        fpr, recall_1, seuils = roc_curve(y_test, probs_test[:,1], pos_label=1)        
        df_seuil_recall1 = pd.DataFrame({'Recall_1': recall_1, 'Seuils_proba' : seuils})

        p80 = df_seuil_recall1.loc[df_seuil_recall1['Recall_1'] <= 0.8].tail(1)['Seuils_proba'].iloc[0]
        return p80
    
    def r40_score(probs_test):
        
        # DataFrame df_seuil_recall1 
        fpr, recall_1, seuils = roc_curve(y_test, probs_test[:,1], pos_label=1)        
        df_seuil_recall1 = pd.DataFrame({'Recall_1': recall_1, 'Seuils_proba' : seuils})

        r40 = df_seuil_recall1.loc[df_seuil_recall1['Seuils_proba'] <= 0.4].head(1)['Recall_1'].iloc[0]
        return r40
    
# Courbe lift-cumulée
######################################################################################################
    import matplotlib.pyplot as plt
    import scikitplot as skplt

    # Courbe tracée
    ##############################
    def courbe_lift_cum(probs_test):
        # Tracer la courbe de gain cumulative
        fig, ax = plt.subplots(figsize=(10, 6))
        skplt.metrics.plot_cumulative_gain(y_test, probs_test, ax=ax)

        # Récupérer les valeurs de la courbe de gain
        x_values = ax.lines[1].get_xdata()
        y_values = ax.lines[1].get_ydata()

        # Créer un DataFrame pour stocker les valeurs de la courbe de gain
        df_gain = pd.DataFrame({'% de la clientèle': x_values, '% des souscripteurs atteint': y_values})

        x_25 = df_gain['% de la clientèle'].loc[df_gain['% de la clientèle'] >= 0.25].head(1).iloc[0]
        y_25 = df_gain['% des souscripteurs atteint'].loc[df_gain['% de la clientèle'] >= 0.25].head(1).iloc[0]

        x_50 = df_gain['% de la clientèle'].loc[df_gain['% de la clientèle'] >= 0.5].head(1).iloc[0]
        y_50 = df_gain['% des souscripteurs atteint'].loc[df_gain['% de la clientèle'] >= 0.5].head(1).iloc[0]

        x_75 = df_gain['% de la clientèle'].loc[df_gain['% de la clientèle'] >= 0.75].head(1).iloc[0]
        y_75 = df_gain['% des souscripteurs atteint'].loc[df_gain['% de la clientèle'] >= 0.75].head(1).iloc[0]


        # Point 25%
        plt.plot([x_25],[y_25], marker = 'o',color = 'yellow', markersize = 10, label = f'En appelant 25% de la clientèle, {round(y_25,2)}% des souscripteurs sont ciblés')

        plt.plot([x_25, x_25], [0, y_25], linestyle='-.', color='yellow',linewidth = 1.5)
        plt.plot([0, x_25], [y_25, y_25], linestyle='-.', color='yellow',linewidth = 1.5)

        plt.text(x_25+0.035, 0.015, f'x25%', fontsize=10, ha='center', color='black')
        plt.text(0.005, y_25-0.03, f'y25%={round(y_25,2)}', fontsize=10, va='center', color='black')

        # Point 50%
        plt.plot([x_50],[y_50], marker = 'o',color = 'green', markersize = 10, label = f'En appelant 50% de la clientèle, {round(y_50,2)}% des souscripteurs sont ciblés')

        plt.plot([x_50, x_50], [0, y_50], linestyle='-.', color='green',linewidth = 1.5)
        plt.plot([0, x_50], [y_50, y_50], linestyle='-.', color='green',linewidth = 1.5)

        plt.text(x_50+0.035, 0.015, f'x50%', fontsize=10, ha='center', color='black')
        plt.text(0.005, y_50-0.03, f'y50%={round(y_50,2)}', fontsize=10, va='center', color='black')

        # Point 75%
        plt.plot([x_75],[y_75], marker = 'o',color = 'purple', markersize = 10, label = f'En appelant 75% de la clientèle, {round(y_75,2)}% des souscripteurs sont ciblés')

        plt.plot([x_75, x_75], [0, y_75], linestyle='-.', color='purple',linewidth = 1.5)
        plt.plot([0, x_75], [y_75, y_75], linestyle='-.', color='purple',linewidth = 1.5)

        plt.text(x_75+0.035, 0.015, f'x75%', fontsize=10, ha='center', color='black')
        plt.text(0.005, y_75-0.03, f'y75%={round(y_75,2)}', fontsize=10, va='center', color='black')
        plt.grid(False)

        st.pyplot(fig)

        exp_courbe1 = f"En ciblant **25%** de la clientèle, nous atteignons un taux de souscription de **{round(y_25,3)*100}%**"
        st.write(exp_courbe1)
 
        exp_courbe2 = f"En ciblant **50%** de la clientèle, nous atteignons un taux de souscription de **{round(y_50,3)*100}%**"
        st.write(exp_courbe2)

        exp_courbe3 = f"En ciblant **75%** de la clientèle, nous atteignons un taux de souscription de **{round(y_75,3)*100}%**"
        st.write(exp_courbe3)
        st.write("\n\n")        

        

# Partie Streamlit
############################################################################################################
############################################################################################################
  
    st.write("")      
    st.write("##### Choisir la dimension du jeu de données")
    # Dimension du Dataset
    choix_dim_dataset = [0, 1, 2, 3, 4, 5]
    option_dim_dataset = st.selectbox("Dimension du jeu de données", choix_dim_dataset)

    # Partie code - Xtrain, X_train_1 etc
    #####################################

    X_test_1 = X_test.drop(['default','job_housemaid','job_student','job_entrepreneur','job_unemployed',
                        'job_retired','job_self-employed','job_services',
                        'job_blue-collar','job_technician','job_management'],axis = 1)
    X_train_1 = X_train.drop(['default','job_housemaid','job_student','job_entrepreneur','job_unemployed',
                        'job_retired','job_self-employed','job_services',
                        'job_blue-collar','job_technician','job_management'],axis = 1)
    
    X_test_2 = X_test_1.drop(['loan'],axis = 1)
    X_train_2 = X_train_1.drop(['loan'],axis = 1)

    X_test_3 = X_test_2.drop(['marital_single','marital_married'],axis = 1)
    X_train_3 = X_train_2.drop(['marital_single','marital_married'],axis = 1)

    X_test_4 = X_test_3.drop('education',axis = 1)
    X_train_4 = X_train_3.drop('education',axis = 1)

    X_test_5 = X_test_4.drop('previous',axis = 1)
    X_train_5 = X_train_4.drop('previous',axis = 1)

    #####################################
    
    if option_dim_dataset == 0:
        if st.checkbox("*Jeu de données d'entraînement après réduction*") :
            st.dataframe(X_train)

    if option_dim_dataset == 1:
        if st.checkbox("*Jeu de données d'entraînement après réduction*") :
            st.dataframe(X_train_1)

    if option_dim_dataset == 2:
        if st.checkbox("*Jeu de données d'entraînement après réduction*") :
            st.dataframe(X_train_2)

    if option_dim_dataset == 3:
        if st.checkbox("*Jeu de données d'entraînement après réduction*") :
            st.dataframe(X_train_3)

    if option_dim_dataset == 4:
        if st.checkbox("*Jeu de données d'entraînement après réduction*") :
            st.dataframe(X_train_4)

    if option_dim_dataset == 5:
        if st.checkbox("*Jeu de données d'entraînement après réduction*") :
            st.dataframe(X_train_5)
      
    st.write("")  
    # Choix du modèle
    st.write("##### Choisir le modèle de Lachine Learning")
    choix_model = [None, 'Random Forest', 'Logistic Regression', 'Gradient Boosting']
    option_model = st.selectbox('Choix du modèle', choix_model)

    st.write("") 
    # Choix du seuil de classification
    st.write("##### Choisir le seuil de classification")
    option_seuil = st.slider('Choix du seuil de classification',0.3,0.7,0.4)

# Fin des premiers choix
############################################################################################################
    if option_model == None:
        st.write("En attente du choix d'un modèle")
    
    if option_model != None:
        st.write(f'Le modèle choisi est **{option_model}**',f", la dimension du jeu de donnée: **{option_dim_dataset}**",f", le seuil de classification: **{option_seuil*100}**%") 
        st.write("")
    
#Librairie modèles
############################################################################################################
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier

#dim_dataset = 0
###############################################################
    if option_dim_dataset == 0:

        #Logistic Regression
        ################################################
        ################################################  
        if option_model == 'Logistic Regression':
            st.write("")             
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_penalty = ['l2',None]
            option_penalty = st.selectbox('penalty', choix_penalty)

            choix_C = [0.5,1.0,5.0,10.0]
            option_C = st.selectbox('C', choix_C)

            choix_max_iter = [50,100,150,200]
            option_max_iter = st.selectbox('max_iter', choix_max_iter)

            if st.button("Lancer le modèle - Régression Logistique"):
            
                # Entraînement du modèle
                clf = LogisticRegression(penalty = option_penalty, C = option_C ,max_iter = option_max_iter)
                clf.fit(X_train, y_train)         
            
                probs_train  = clf.predict_proba(X_train)
                probs_test  = clf.predict_proba(X_test)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                
                ###############
                ###############
        
        #Random Forest
        ################################################
        ################################################
        if option_model == 'Random Forest':

            st.write("")
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_max_depth = [5,10,15,20,25,30]
            option_max_depth = st.selectbox('max_depth', choix_max_depth)       

            choix_n_estimators = [50,100,150]
            option_n_estimators = st.selectbox('n_estimators', choix_n_estimators)

            choix_min_samples_leaf = [1,2,5]
            option_min_samples_leaf = st.selectbox('min_samples_leaf', choix_min_samples_leaf)

            choix_criterion = ['gini', 'entropy']
            option_criterion = st.selectbox('criterion', choix_criterion)


            if st.button("Lancer le modèle - Random Forest"):

                # Entraînement du modèle
                clf = RandomForestClassifier(random_state=42,max_depth = option_max_depth, n_estimators = option_n_estimators ,min_samples_leaf = option_min_samples_leaf,criterion=option_criterion)
                clf.fit(X_train, y_train)         
            
                probs_train  = clf.predict_proba(X_train)
                probs_test  = clf.predict_proba(X_test)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                
                ###############
                ###############        



        #Gradient Boosting
        ################################################
        ################################################
        if option_model == 'Gradient Boosting':
        
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_subsample = [0.5,0.75,1]
            option_subsample = st.selectbox('subsample', choix_subsample)       

            choix_learning_rate = [0.05,0.075,0.1,0.125,0.15]
            option_learning_rate = st.selectbox('learning_rate', choix_learning_rate)

            choix_n_estimators2 = [50,100,150,200]
            option_n_estimators2 = st.selectbox('n_estimators', choix_n_estimators2)

            choix_max_depth2 = [2,3,4,5]
            option_max_depth2 = st.selectbox('max_depth', choix_max_depth2)


            if st.button("Lancer le modèle - Gradient Boosting"):

                # Entraînement du modèle
                clf = GradientBoostingClassifier(random_state = 42,max_depth = option_max_depth2, subsample = option_subsample ,learning_rate= option_learning_rate,n_estimators=option_n_estimators2)
                clf.fit(X_train, y_train)         
                
                probs_train  = clf.predict_proba(X_train)
                probs_test  = clf.predict_proba(X_test)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                    
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                    
                    ###############
                    ###############  


#dim_dataset = 1
###############################################################
    if option_dim_dataset == 1:
        
        #Logistic Regression
        ################################################
        ################################################  
        if option_model == 'Logistic Regression':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_penalty = ['l2',None]
            option_penalty = st.selectbox('penalty', choix_penalty)

            choix_C = [0.5,1.0,5.0,10.0]
            option_C = st.selectbox('C', choix_C)

            choix_max_iter = [50,100,150,200]
            option_max_iter = st.selectbox('max_iter', choix_max_iter)


            if st.button("Lancer le modèle - Régression Logistique"):
                
                # Entraînement du modèle
                clf = LogisticRegression(penalty = option_penalty, C = option_C ,max_iter = option_max_iter)
                clf.fit(X_train_1, y_train)         
                
                probs_train  = clf.predict_proba(X_train_1)
                probs_test  = clf.predict_proba(X_test_1)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                    
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                    
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                    
                ###############
                ###############



        #Random Forest
        ################################################
        ################################################
        if option_model == 'Random Forest':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_max_depth = [5,10,15,20,25,30]
            option_max_depth = st.selectbox('max_depth', choix_max_depth)       

            choix_n_estimators = [50,100,150]
            option_n_estimators = st.selectbox('n_estimators', choix_n_estimators)

            choix_min_samples_leaf = [1,2,5]
            option_min_samples_leaf = st.selectbox('min_samples_leaf', choix_min_samples_leaf)

            choix_criterion = ['gini', 'entropy']
            option_criterion = st.selectbox('criterion', choix_criterion)


            if st.button("Lancer le modèle - Random Forest"):

                # Entraînement du modèle
                clf = RandomForestClassifier(random_state = 42,max_depth = option_max_depth, n_estimators = option_n_estimators ,min_samples_leaf = option_min_samples_leaf,criterion=option_criterion)
                clf.fit(X_train_1, y_train)         
            
                probs_train  = clf.predict_proba(X_train_1)
                probs_test  = clf.predict_proba(X_test_1)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############
        
        
        #Gradient Boosting
        ################################################
        ################################################
        if option_model == 'Gradient Boosting':
        
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_subsample = [0.5,0.75,1]
            option_subsample = st.selectbox('subsample', choix_subsample)       

            choix_learning_rate = [0.05,0.075,0.1,0.125,0.15]
            option_learning_rate = st.selectbox('learning_rate', choix_learning_rate)

            choix_n_estimators2 = [50,100,150,200]
            option_n_estimators2 = st.selectbox('n_estimators', choix_n_estimators2)

            choix_max_depth2 = [2,3,4,5]
            option_max_depth2 = st.selectbox('max_depth', choix_max_depth2)


            if st.button("Lancer le modèle - Gradient Boosting"):

                # Entraînement du modèle
                clf = GradientBoostingClassifier(random_state = 42,max_depth = option_max_depth2, subsample = option_subsample ,learning_rate= option_learning_rate,n_estimators=option_n_estimators2)
                clf.fit(X_train_1, y_train)         
            
                probs_train  = clf.predict_proba(X_train_1)
                probs_test  = clf.predict_proba(X_test_1)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                
                ###############
                ############### 


#dim_dataset = 2
###############################################################
    if option_dim_dataset == 2:

        #Logistic Regression
        ################################################
        ################################################  
        if option_model == 'Logistic Regression':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_penalty = ['l2',None]
            option_penalty = st.selectbox('penalty', choix_penalty)

            choix_C = [0.5,1.0,5.0,10.0]
            option_C = st.selectbox('C', choix_C)

            choix_max_iter = [50,100,150,200]
            option_max_iter = st.selectbox('max_iter', choix_max_iter)

            if st.button("Lancer le modèle - Régression Logistique"):
            
                # Entraînement du modèle
                clf = LogisticRegression(penalty = option_penalty, C = option_C ,max_iter = option_max_iter)
                clf.fit(X_train_2, y_train)         
            
                probs_train  = clf.predict_proba(X_train_2)
                probs_test  = clf.predict_proba(X_test_2)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############


        #Random Forest
        ################################################
        ################################################
        if option_model == 'Random Forest':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_max_depth = [5,10,15,20,25,30]
            option_max_depth = st.selectbox('max_depth', choix_max_depth)       

            choix_n_estimators = [50,100,150]
            option_n_estimators = st.selectbox('n_estimators', choix_n_estimators)

            choix_min_samples_leaf = [1,2,5]
            option_min_samples_leaf = st.selectbox('min_samples_leaf', choix_min_samples_leaf)

            choix_criterion = ['gini', 'entropy']
            option_criterion = st.selectbox('criterion', choix_criterion)


            if st.button("Lancer le modèle - Random Forest"):

                # Entraînement du modèle
                clf = RandomForestClassifier(random_state = 42,max_depth = option_max_depth, n_estimators = option_n_estimators ,min_samples_leaf = option_min_samples_leaf,criterion=option_criterion)
                clf.fit(X_train_2, y_train)         
            
                probs_train  = clf.predict_proba(X_train_2)
                probs_test  = clf.predict_proba(X_test_2)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############

        
        
        #Gradient Boosting
        ################################################
        ################################################
        if option_model == 'Gradient Boosting':
        
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_subsample = [0.5,0.75,1]
            option_subsample = st.selectbox('subsample', choix_subsample)       

            choix_learning_rate = [0.05,0.075,0.1,0.125,0.15]
            option_learning_rate = st.selectbox('learning_rate', choix_learning_rate)

            choix_n_estimators2 = [50,100,150,200]
            option_n_estimators2 = st.selectbox('n_estimators', choix_n_estimators2)

            choix_max_depth2 = [2,3,4,5]
            option_max_depth2 = st.selectbox('max_depth', choix_max_depth2)


            if st.button("Lancer le modèle - Gradient Boosting"):

                # Entraînement du modèle
                clf = GradientBoostingClassifier(random_state = 42,max_depth = option_max_depth2, subsample = option_subsample ,learning_rate= option_learning_rate,n_estimators=option_n_estimators2)
                clf.fit(X_train_2, y_train)         
            
                probs_train  = clf.predict_proba(X_train_2)
                probs_test  = clf.predict_proba(X_test_2)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                
                ###############
                ############### 


#dim_dataset = 3
###############################################################
    if option_dim_dataset == 3:


        #Logistic Regression
        ################################################
        ################################################  
        if option_model == 'Logistic Regression':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_penalty = ['l2',None]
            option_penalty = st.selectbox('penalty', choix_penalty)

            choix_C = [0.5,1.0,5.0,10.0]
            option_C = st.selectbox('C', choix_C)

            choix_max_iter = [50,100,150,200]
            option_max_iter = st.selectbox('max_iter', choix_max_iter)

            if st.button("Lancer le modèle - Régression Logistique"):
            
                # Entraînement du modèle
                clf = LogisticRegression(penalty = option_penalty, C = option_C ,max_iter = option_max_iter)
                clf.fit(X_train_3, y_train)         
            
                probs_train  = clf.predict_proba(X_train_3)
                probs_test  = clf.predict_proba(X_test_3)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############             


        #Random Forest
        ################################################
        ################################################
        if option_model == 'Random Forest':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_max_depth = [5,10,15,20,25,30]
            option_max_depth = st.selectbox('max_depth', choix_max_depth)       

            choix_n_estimators = [50,100,150]
            option_n_estimators = st.selectbox('n_estimators', choix_n_estimators)

            choix_min_samples_leaf = [1,2,5]
            option_min_samples_leaf = st.selectbox('min_samples_leaf', choix_min_samples_leaf)

            choix_criterion = ['gini', 'entropy']
            option_criterion = st.selectbox('criterion', choix_criterion)


            if st.button("Lancer le modèle - Random Forest"):

                # Entraînement du modèle
                clf = RandomForestClassifier(random_state = 42,max_depth = option_max_depth, n_estimators = option_n_estimators ,min_samples_leaf = option_min_samples_leaf,criterion=option_criterion)
                clf.fit(X_train_3, y_train)         
            
                probs_train  = clf.predict_proba(X_train_3)
                probs_test  = clf.predict_proba(X_test_3)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############

        
        
        #Gradient Boosting
        ################################################
        ################################################
        if option_model == 'Gradient Boosting':
        
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_subsample = [0.5,0.75,1]
            option_subsample = st.selectbox('subsample', choix_subsample)       

            choix_learning_rate = [0.05,0.075,0.1,0.125,0.15]
            option_learning_rate = st.selectbox('learning_rate', choix_learning_rate)

            choix_n_estimators2 = [50,100,150,200]
            option_n_estimators2 = st.selectbox('n_estimators', choix_n_estimators2)

            choix_max_depth2 = [2,3,4,5]
            option_max_depth2 = st.selectbox('max_depth', choix_max_depth2)


            if st.button("Lancer le modèle - Gradient Boosting"):

                # Entraînement du modèle
                clf = GradientBoostingClassifier(random_state = 42,max_depth = option_max_depth2, subsample = option_subsample ,learning_rate= option_learning_rate,n_estimators=option_n_estimators2)
                clf.fit(X_train_3, y_train)         
            
                probs_train  = clf.predict_proba(X_train_3)
                probs_test  = clf.predict_proba(X_test_3)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                
                ###############
                ############### 




#dim_dataset = 4
###############################################################
    if option_dim_dataset == 4:

         #Logistic Regression
        ################################################
        ################################################  
        if option_model == 'Logistic Regression':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_penalty = ['l2',None]
            option_penalty = st.selectbox('penalty', choix_penalty)

            choix_C = [0.5,1.0,5.0,10.0]
            option_C = st.selectbox('C', choix_C)

            choix_max_iter = [50,100,150,200]
            option_max_iter = st.selectbox('max_iter', choix_max_iter)

            if st.button("Lancer le modèle - Régression Logistique"):
            
                # Entraînement du modèle
                clf = LogisticRegression(penalty = option_penalty, C = option_C ,max_iter = option_max_iter)
                clf.fit(X_train_4, y_train)         
            
                probs_train  = clf.predict_proba(X_train_4)
                probs_test  = clf.predict_proba(X_test_4)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############   



        #Random Forest
        ################################################
        ################################################
        if option_model == 'Random Forest':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_max_depth = [5,10,15,20,25,30]
            option_max_depth = st.selectbox('max_depth', choix_max_depth)       

            choix_n_estimators = [50,100,150]
            option_n_estimators = st.selectbox('n_estimators', choix_n_estimators)

            choix_min_samples_leaf = [1,2,5]
            option_min_samples_leaf = st.selectbox('min_samples_leaf', choix_min_samples_leaf)

            choix_criterion = ['gini', 'entropy']
            option_criterion = st.selectbox('criterion', choix_criterion)


            if st.button("Lancer le modèle - Random Forest"):

                # Entraînement du modèle
                clf = RandomForestClassifier(random_state = 42,max_depth = option_max_depth, n_estimators = option_n_estimators ,min_samples_leaf = option_min_samples_leaf,criterion=option_criterion)
                clf.fit(X_train_4, y_train)         
            
                probs_train  = clf.predict_proba(X_train_4)
                probs_test  = clf.predict_proba(X_test_4)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############

        
        
        #Gradient Boosting
        ################################################
        ################################################
        if option_model == 'Gradient Boosting':
        
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_subsample = [0.5,0.75,1]
            option_subsample = st.selectbox('subsample', choix_subsample)       

            choix_learning_rate = [0.05,0.075,0.1,0.125,0.15]
            option_learning_rate = st.selectbox('learning_rate', choix_learning_rate)

            choix_n_estimators2 = [50,100,150,200]
            option_n_estimators2 = st.selectbox('n_estimators', choix_n_estimators2)

            choix_max_depth2 = [2,3,4,5]
            option_max_depth2 = st.selectbox('max_depth', choix_max_depth2)


            if st.button("Lancer le modèle - Gradient Boosting"):

                # Entraînement du modèle
                clf = GradientBoostingClassifier(random_state = 42,max_depth = option_max_depth2, subsample = option_subsample ,learning_rate= option_learning_rate,n_estimators=option_n_estimators2)
                clf.fit(X_train_4, y_train)         
            
                probs_train  = clf.predict_proba(X_train_4)
                probs_test  = clf.predict_proba(X_test_4)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite_v2(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                
                ###############
                ############### 



#dim_dataset = 5
###############################################################
    if option_dim_dataset == 5:
            
        #Logistic Regression
        ################################################
        ################################################  
        if option_model == 'Logistic Regression':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_penalty = ['l2',None]
            option_penalty = st.selectbox('penalty', choix_penalty)

            choix_C = [0.5,1.0,5.0,10.0]
            option_C = st.selectbox('C', choix_C)

            choix_max_iter = [50,100,150,200]
            option_max_iter = st.selectbox('max_iter', choix_max_iter)

            if st.button("Lancer le modèle - Régression Logistique"):
            
                # Entraînement du modèle
                clf = LogisticRegression(penalty = option_penalty, C = option_C ,max_iter = option_max_iter)
                clf.fit(X_train_5, y_train)         
            
                probs_train  = clf.predict_proba(X_train_5)
                probs_test  = clf.predict_proba(X_test_5)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ############### 



        #Random Forest
        ################################################
        ################################################
        if option_model == 'Random Forest':
            
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_max_depth = [5,10,15,20,25,30]
            option_max_depth = st.selectbox('max_depth', choix_max_depth)       

            choix_n_estimators = [50,100,150]
            option_n_estimators = st.selectbox('n_estimators', choix_n_estimators)

            choix_min_samples_leaf = [1,2,5]
            option_min_samples_leaf = st.selectbox('min_samples_leaf', choix_min_samples_leaf)

            choix_criterion = ['gini', 'entropy']
            option_criterion = st.selectbox('criterion', choix_criterion)


            if st.button("Lancer le modèle - Random Forest"):

                # Entraînement du modèle
                clf = RandomForestClassifier(random_state = 42,max_depth = option_max_depth, n_estimators = option_n_estimators ,min_samples_leaf = option_min_samples_leaf,criterion=option_criterion)
                clf.fit(X_train_5, y_train)         
            
                probs_train  = clf.predict_proba(X_train_5)
                probs_test  = clf.predict_proba(X_test_5)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite(probs_test)

                # Courbe lift-cumulée
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write('En attente du choix d un modèle')
                
                ###############
                ###############

        
        
        #Gradient Boosting
        ################################################
        ################################################
        if option_model == 'Gradient Boosting':
        
            st.write("##### Choisir les hyperparamètres du modèle")

            choix_subsample = [0.5,0.75,1]
            option_subsample = st.selectbox('subsample', choix_subsample)       

            choix_learning_rate = [0.05,0.075,0.1,0.125,0.15]
            option_learning_rate = st.selectbox('learning_rate', choix_learning_rate)

            choix_n_estimators2 = [50,100,150,200]
            option_n_estimators2 = st.selectbox('n_estimators', choix_n_estimators2)

            choix_max_depth2 = [2,3,4,5]
            option_max_depth2 = st.selectbox('max_depth', choix_max_depth2)

            if st.button("Lancer le modèle - Gradient Boosting"):

                # Entraînement du modèle
                clf = GradientBoostingClassifier(random_state = 42,max_depth = option_max_depth2, subsample = option_subsample ,learning_rate= option_learning_rate,n_estimators=option_n_estimators2)
                clf.fit(X_train_5, y_train)         
            
                probs_train  = clf.predict_proba(X_train_5)
                probs_test  = clf.predict_proba(X_test_5)

                y_pred_train = np.where(probs_train[:,1]> option_seuil,1,0)
                y_pred_test = np.where(probs_test[:,1]> option_seuil,1,0)            
                
                # Tableau Récapitulatif
                a = accuracy_score(y_test,y_pred_test)
                b = overfit_score(y_pred_train,y_pred_test)
                c = recall1_score(y_pred_test)
                d = compute_score(y_pred_test)
                df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                st.write("")                
                st.write("##### Tableau récapitulatif des résultats")  
                st.dataframe(df_score)

                # Matrice de confusion 
                conf_matrix = conf_matrix(y_pred_test)
                st.write("")                
                st.write("##### Matrice de confusion")
                st.text(conf_matrix)

                # Courbe sensibilité en fonction du seuil de classification
                st.write("")
                st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
                courbe_sensibilite_v2(probs_test)

                # Courbe lift-cumulée
                st.write("")
                st.write("##### Courbe lift_cumulée - capacité de ciblage")
                courbe_lift_cum(probs_test)
                st.write("")

                # Réinitialisation
                if st.button("Réinitialiser"):
                    st.write("Réinitialisé")
                
                ###############
                ############### 

# Partie 4
############################################################################################################
############################################################################################################
    st.write("")
    st.write("#### Notre  méthode de sélection du modèle adequat, par itérations")

    if st.checkbox("Afficher le schéma de notre méthode de sélection"):
        st.image("methode_iterative.png", use_column_width=True)


# Partie 5
############################################################################################################
############################################################################################################
    st.write("")
    st.write("#### Modèle retenu")

    def contenu_partie_5():
        model_retenu = """
        Après application de la méthode d'itération sur les 3 types de modèles Régression logistique - Random Forest - Gradient Boosting, voici les 
        caractèristiques du modèle retenu:\n
        - Le modèle est de type Random Forest\n
        - Il s'entraîne sur la totalité du jeu d'entraînement après pré-traitement (rappel : hors des données de la campagne actuelle)\n
        - Sa profondeur d'arbre est de 5\n
        - Son nombre d'estimateurs est de 150\n
        - L'échantillon minimum de feuille est de 5
        """

        st.write(model_retenu)

        if st.button("Lancer le modèle retenu"):
            
            # Entraînement du modèle
            best_clf = RandomForestClassifier(max_depth = 5, n_estimators = 150, min_samples_leaf = 5,random_state = 42)
            best_clf.fit(X_train, y_train)         
            
            probs_train_best  = best_clf.predict_proba(X_train)
            probs_test_best  = best_clf.predict_proba(X_test)

            y_pred_train_best = np.where(probs_train_best[:,1]> option_seuil,1,0)
            y_pred_test_best = np.where(probs_test_best[:,1]> option_seuil,1,0)            
                
            # Tableau Récapitulatif
            a = accuracy_score(y_test,y_pred_test_best)
            b = overfit_score(y_pred_train_best,y_pred_test_best)
            c = recall1_score(y_pred_test_best)
            d = compute_score(y_pred_test_best)
            df_score = pd.DataFrame({"Accuracy": [f'{round(a*100,2)}%'],"Overfitting": [b], "Score de rappel classe 1":[f'{round(c*100,2)}%'],"Metric_balance":[d]}, index = ["Scores"])
                
            st.write("##### Tableau récapitulatif des résultats")  
            st.dataframe(df_score)

            # Matrice de confusion 
            conf_matrix_best = conf_matrix(y_pred_test_best)
            st.write("##### Matrice de confusion")
            st.text(conf_matrix_best)

            # Courbe sensibilité en fonction du seuil de classification
            st.write("##### Courbe de la sensibilité en fonction du seuil de classification")
            courbe_sensibilite(probs_test_best)

            # Courbe lift-cumulée
            st.write("##### Courbe lift-cumulée")
            courbe_lift_cum(probs_test_best)

            if st.button("Refermer"):
                st.write('En attente du choix d un modèle')

    if st.checkbox("Afficher le modèle retenu et ses indicateurs de performance"):
        contenu_partie_5()

############################################################################################################
############################################################################################################

# Page 5 Interprétation des modèles

############################################################################################################
############################################################################################################

if page == pages[5] : 
    st.write("### 5. Interprétations des modèles")

    st.write("#### Analyse de l'influence des variables - modèle RandomForest retenu après itération")

    st.write("Regardons l'importance du rôle qu'a joué chaque variable dans les choix de prédiction du modèle Random Forest choisi.")
    
    st.image('feats_importance.png', use_column_width=True)

    interpretation_rf1 = """
    Pour prendre sa décision de prédiction, le modèle s'appuie donc principalement sur ces données client:\n
    - L'âge du client.
    - A-t-il un prêt immobilier ?
    - Le montant de son solde bancaire.
    """
    st.write(interpretation_rf1)

    st.write("#### Analyse de l'influence des variables - application d'un modèle SHAP sur un modèle Gradient Boosting")


    shap_explication = """
    Le modèle de SHAP se base sur la théorie des jeux. Il consiste à calculer la contribution marginale de chaque variable sur la prédiction 
    pour un individu. En d’autres termes, il répond à la question suivante : “Pour un individu A, comment et à quel point est-ce que la variable
    X influence la prédiction ?”
    Concrètement, après avoir utilisé le modèle de Shap, on obtient un dataframe de même dimension que le jeu d’entraînement,
    mais les valeurs sont remplacées par les shap-values. Pour une instance, les shap-values indiquent alors la contribution marginale
    de chaque variable. En les additionnant, on tombe sur la prédictions de l’instance concernée. 
    """
    if st.checkbox("Qu'est ce qu'un modèle de SHAP?"):
        st.write(shap_explication)
    
    st.write("Après application du modèle de SHAP au modèle Gradient Boosting, nous obtenons ces résultats:")

    st.image('shap_2.png', width = 600)

    shap_exp3 = """
    En plus de classer les variables de la plus à la moins influente, le beeswarmplot permet de voir la nature de l’influence 
    de chaque variable pour toutes les instances du jeu. Un point correspond donc à une  valeur, par exemple, la valeur de la variable month 
    pour l’individu 10. 
    Plus un point est à gauche, plus la variable pour l’instance en question tend vers la classe 0. 
    À l’inverse, plus un point est à droite, plus la variable pour l’instance en question influe positivement sur le dépôt à terme. 
    La couleur donne une idée de la modalité dont il s’agit. Plus la modalité concernée est proche de 0 (après normalisation), plus la 
    couleur sera bleue. Plus la modalité est élevée, plus la couleur sera rouge. Par exemple, pour la variable mois qui contient 12 modalités de 
    0 à 12, les premiers mois de l’année seront les plus proches du bleu, tandis que les derniers mois seront les plus proches du rouge.
    """
    if st.checkbox('Comment lire le graphe beeswarmplot? - ci-dessus'):
        st.write(shap_exp3)

    shap_exp4 = """
    Que peut-on tirer du graphe ci-dessus?\n
    - On constate donc que pour le modèle GradientBoost (qui a des résultats très similaires au Random Forrest choisi), lorsque le moyen de *contact* est inconnu, la variable influe négativement sur 
    le dépôt à terme. À l’inverse, lorsqu’il est connu, il influe positivement sur le dépôt à terme mais de manière un peu plus intense. 
    Il est toutefois difficile d’analyser l’influence de cette variable, puisque nous ne savons pas pourquoi certaines valeurs étaient manquantes.\n
    - En ce qui concerne *Month*, on constate que la distribution est assez hétérogène. Même si elle tend plus vers le positif. 
    On voit toutefois une regroupement importants d’individus violets dans le négatif, qui correspondent aux mois de Mai et Juin. 
    Cela fait sens puisque notre travail de visualisation de donnée avait déjà montré un taux élevé d’échec sur ces mois, 
    où le nombre de contacts avait explosé.\n
    - Concernant *housing*, la distribution prend une nature différente, plus homogène. En effet, les individus en rouge sont nettement 
    dans le négatif, et les individus en bleu nettement dans le positif. Encore une fois, l’interprétation du GradientBoosting fait écho 
    aux observations faites précédemment : la présence d’un prêt immobilier représente un gros frein au dépôt à terme.
    Dans une certaine mesure, ce constat est le même avec le prêt à la consommation (loan).\n
    - On peut également voir que d’après le modèle, les individus avec des *équilibres des comptes* plus faibles auront moins de chance de souscrire 
    un dépôt à terme.
    - Finalement la variable *âge* vient également confirmer une observation précédente : les jeunes et les seniors sont des cibles intéressantes.
    Cela dit, on constate également la présence de ces segments du côté négatif du graphe.\n 
    Afin d’y voir plus clair, on peut utiliser un autre type de plot :
     """
    st.write(shap_exp4)

    st.image("shap_3.png", use_column_width=True)

    shap_exp5 = """
    Ce forceplot permet de visualiser les influences au niveau global. Dans ce cas, le graphe se focalise sur les shap-values de chaque instance pour la colonne age. Ici le graphe est classé de l’individu le plus jeune au plus âgé. 
    Attention, la couleur ne se réfère pas à la valeur de la variable mais à son importance (selon le modèle 'SHAP') concernant la variable cible, donc le rouge correspond aux instances avec une probabilité haute de souscrire un dépôt.\n
    Cela confirme donc que les catégories les plus âgées et les plus jeunes sont plus aptes à souscrire au dépôt à terme.
    """
    if st.checkbox("Analyse du graphe"):
        st.write(shap_exp5)       
    st.write("###### Conclusion")
    shap_exp6 = """
    Le modèle de SHAP est idéal pour interpréter des modèles complexes de type “boîte noire”, difficilement interprétables par nature. 
    En l’occurrence, SHAP semble indiquer que les prédictions du GradientBoost correspondent à nos observations sur le jeu de données initial : 
    la situation socio-économique des individus est clé pour la souscription d’un dépôt à terme.
    """
    st.write(shap_exp6) 

if page == pages[6]:
    
  st.write("### 6. Pour aller plus loin : constitution de clusters pour améliorer l'efficacité globale des campagnes")

  if st.checkbox("Intérêt de la réduction de dimension pour réaliser des clusters"):
    st.write("Les clusters sont plus intéressants lorsque réalisés sur des données dont les composantes ont été réduites. Les opérations de réduction de dimension consistent à conserver le plus d’information sur le moins d’axes possible, idéalement 2 pour pouvoir être représentées graphiquement. Par exemple, les informations contenues dans 10 colonnes peuvent se retrouver combinées en 2 colonnes, avec une perte d'information très faible.")

  st.write("L'algorithme TSNE a permis d'opérer la distinction la plus nette entre les groupes d'individus similaires. Il va regrouper ensemble des individus qui ont de fortes probabilités d'avoir des données similaires, et les éloigner des individus suceptibles d'avoir des données différentes. Voici sa représentation sur deux axes, avec un distinction selon que les individus aient souscrit ou non un dépôt à terme :")

  st.image('tsne.png')

  st.write("A noter une plus forte concentration des individus qui ont souscrit en haut et à droite (points orange). En appliquant l'algorithme de Machine Learning non supervisé “Kmeans”, nous obtenons 5 clusters,  en forme de pétales :")

  st.image('clusters.png')

  if st.checkbox("Glossaire : vocabulaire utilisé pour l'analyse"):
    st.write("Nous présenterons plus bas une synthèse des différents clusters. Mais avant, voici quelques définitions sur le vocabulaire que nous allons utiliser :")

    multi = '''
    - *Education primaire* : fin de l'école à 15 ans (équivalent collège)
    - *Education secondaire* : fin de l'école à 18 ans (équivalent lycée)
    - *Education tertiaire* : études supérieures
    - *Disponibilité financière* : il s'agit d'un terme utilisé spécifiquement pour ce travail de recherche. Cela combine à la fois le montant moyen sur les comptes bancaires et la souscription actuelle à des prêts immobiliers et à la consommation.
    - *Deposit* : souscription d’un dépôt à terme
    '''

    st.markdown(multi)
  
  if st.checkbox("Synthèse cluster 1 (*bleu*)"):

    st.image("2_etoiles.png")

    multi = '''
   - Taux de deposit de 33%
   - 97% ont un diplôme du secondaire
   - Plus représentés chez les travailleurs dans les services
   - 100% ont un crédit immobilier
   - 1 171€ dans le compte bancaire en moyenne
    '''

    st.markdown(multi)

    st.write("Un cluster 'intermédiaire', qui a peu de chance de souscrire, car a une disponibilité financière assez faible.")

  if st.checkbox("Synthèse cluster 2 (*rouge*)"):

    st.image("4_etoiles.png")

    multi = '''
   - Taux de deposit de 50%
   - Très forte représentation chez les retraités, les étudiants, et chez les individus sans emplois
   - Très forte représentation chez les plus de 60 ans.
   - 2% ont un crédit immobilier
   - 2 076€ dans le compte bancaire en moyenne (montant le plus fort)
    '''

    st.markdown(multi)

    st.write("Un cluster qui regroupe un nombre important d'individus inactifs : à la fois les étudiants, retraités, sans emplois ; sans pour autant s'y limiter. La disponibilité bancaire est la plus importante parmi les clusters.")

  if st.checkbox("Synthèse cluster 3 (*vert*)"):

    st.image("5_etoiles.png")

    multi = '''
   - Taux de deposit de 60%
   - 96% ont un niveau d’éducation “tertiaire”
   - 55% sont des cadres
   - 1% ont un crédit immobilier
   - 1 741€ dans le compte bancaire en moyenne
    '''

    st.markdown(multi)

    st.write("C'est le cluster qui a le plus fort niveau d'éducation, et les jobs qui en découlent : managers et auto-entrepreneurs. Ils ont également une disponibilité financière importante. La différence de deposit avec le cluster 2 vient du niveau d'éducation, qui donne accès à de meilleurs emplois, salaires et perspectives.")

  if st.checkbox("Synthèse cluster 4 (*violet*)"):

    st.image("1_etoile.png")

    multi = '''
   - Taux de deposit de 31%
   - 36% ont un niveau d’éducation primaire
   - 9% ont fait défaut (seuls parmi les 5 clusters)
   - 70% ont un prêt à la consommation (seuls parmi les 5 clusters)
   - 67% ont un crédit immobilier
   - 782€ dans le compte bancaire en moyenne (montant le plus faible)
    '''

    st.markdown(multi)

    st.write("Couplé au niveau d'éducation le plus faible, il s'agit du cluster montrant les signes les plus importants d'une certaine précarité. La disponibilité financière est faible.")

  if st.checkbox("Synthèse cluster 5 (*orange*)"):

    st.image("2_etoiles.png")

    multi = '''
   - Taux de deposit de 38%
   - 70% ont un niveau d’éducation “tertiaire”
   - 44% sont des cadres
   - 100% ont un crédit immobilier
   - 1 420€ dans le compte bancaire en moyenne
    '''

    st.markdown(multi)

    st.write("Il s'agit de l'autre cluster avec un niveau d'éducation important, bien qu'on le devine moins élevé que le cluster 3. Il se distingue de lui surtout par la souscription d'un crédit immobilier, qui diminue fortement ses disponibilités financières.")

  if st.checkbox("Quelques statistiques descriptives"):
      st.image("c_deposit.png")
      st.write("Les clusters 3 et 2 ont des résultats sur le deposit nettement plus importants que les autres.")
      st.write("")

      st.image("c_education.png")
      text = '''
      Une grande diversité de niveaux d'éducation selon les clusters :

      - Les clusters 3 et 5 sont les plus éduqués, notamment le 3 qui ne compte quasiment que des diplômés	du tertiaire.
      - Le cluster 1 est intermédiaire : il ne compte presque que des diplômés du secondaire.
      - Le cluster 4 a le plus de diplômés du primaire. Le niveau est le moins élevé.
      - Enfin, le cluster 2 compte très peu de diplômés du tertiaire, beaucoup du secondaire, et un nombre non négligeable de diplômés du primaire.
      '''
      st.markdown(text)
      st.write("")

      st.image("c_age.png")
      st.write("Le cluster 2 se distingue dans les âges : plus présent chez les jeunes, et à partir de 50 ans à mesure que l'on vieillit. Ce qui explique le niveau d'éducation 'moindre' parmi ce cluster.")
      st.write("")

      st.image("c_manager.png")
      st.write("Ce graph confirme la forte proportion de cadres parmi les clusters 3 et 5.")
      st.write("")

      st.image("c_housing.png")
      st.write("Le housing est une autre variable discriminante. Les clusters 1 et 5 ont tous un crédit immobilier, alors que les clusters 2 et 3 n'en ont pas. Cette variable permet de distinguer les deux clusters les plus éduqués : le 3 et le 5. La présence d'un crédit immobilier réduit considérablement la disponibilité financière pour souscrire un dépôt à terme.")
      st.write("")

      st.image("c_loan.png")
      st.write("Plus des deux-tiers du cluster 4 ont un prêt à la consommation.")
      st.write("")

      st.image("c_balance.png")
      st.write("Pour finir, le graph montre que le cluster 2 a l'équilibre des comptes moyen le plus important, suivi du cluster 3. Au contraire du cluster 4, et du 1 dans une moindre mesure.")

      if st.checkbox("EN BONUS : des enseignements déjà intégrés dans la stratégie d'appel ?") :
        st.write("Il semble que les équipes sales qui appellent semblent déjà contacter les clients les plus à même de souscrire. Soit 'instinctivement', soit parce que la banque a déjà connaissance des individus qui ont le plus de chances de souscrire et/ou de leur profil.")
        st.image("c_month.png")
        st.write("Nous voyons que, le plupart du temps, les cluster 2 et 3 ont été les plus appelés. Nous constatons également qu'il y a un effet de rattrapage en mai : les individus moins prioritaires ont été contactés “en masse”.")

      st.write("## En résumé")

  multi = '''
 Nous plaçons les clusters sur deux axes :
 - Axe 1 : Disponibilité financière
 - Axe 2 : Niveau d'éducation
 '''
  
  st.markdown(multi)

  st.image('tenseur.png')

  st.write("Cette analyse complète l'interprétation du Machine Learning supervisé. En effet, l'importance prise par les variables housing (crédit immobilier) et âge pour prédire le souscription d'un dépôt à terme laisse entendre que, ce qui est déterminant dans la décision, c'est le stade de vie de l'individu. Où en suis-je dans mes projets ? Est-ce que je dois rembourser mon prêt ? Est-ce que je souhaite acheter bientôt ? Nous comprenons que c’est difficilement mesurable quantitativement. C'est pourquoi les corrélations sont toutes faibles avec la variable deposit. Et pourquoi aucune variable n'est suffisamment déterminante à elle seule pour prédire la souscription. Le Machine Learning supervisé va probabiliser les individus selon leurs stades de vie, notamment s’ils sont “mûrs” pour un dépôt à terme.")

  st.write("Regroupant les individus similaires entre eux, les clusters confirment cette analyse, en apportant quelques couches supplémentaires. Les stades de vie sont, en plus de l'âge, influencés par le niveau d'éducation, qui détermine en partie les perspectives financières d'un individu, ses possibilités d'évolution, et donc sa capacité à se projeter dans l'avenir. Et par la disponibilité financière : en connaissant le montant que j'ai sur mon compte en banque, et en fonction de la présence ou non d'un prêt immobilier (éventuellement d'un prêt à la consommation), puis-je me permettre de souscrire un dépôt à terme ?")

  if st.checkbox("Que faire d'un point de vue opérationnel ?"):

    multi = '''
    Nos recommandations :
   - Utiliser le modèle prédictif du Machine Learning supervisé pour de futures campagnes, et prioriser les individus avec le taux de prédiction de souscription le plus important.
   - Ces individus seront plus présents dans les clusters 3 et 2. Ils vont concentrer la plupart des souscripteurs, et ceux qui auront le compte en banque le plus rempli.
   - Réaliser des entretiens qualitatifs avec ces deux profils pour comprendre leurs stades de vie et leurs mentalités, ce qui sera plus facile pour réaliser des argumentaires à même de les convaincre, et des campagnes de communication percutantes.
   - Utiliser ces arguments lors des campagnes de phoning.
   - Réaliser des campagnes de communication en amont pour faciliter le travail au téléphone, pourquoi pas en adaptant le message selon le cluster grâce à la capacité de ciblage du digital.
    '''

    st.markdown(multi)

  if st.checkbox("Conclusion"):

    multi = '''
    Le projet nous a permis de mettre en oeuvre les compétences apprises lors de la formation. Il souligne l'ampleur des possibilités offertes par l'analyse de larges quantités de données :
   - La réalisation de modèles prédictifs grâce au Machine Learning
   - La réduction de dimension pour avoir une vision synthétique des données
   - La constitution de clusters pour avoir une compréhension "par groupe distincts" du jeu de	données, qui permet notamment de réaliser des personae.
   - La constatation de tendances, de phénomènes (incluant que la découverte de signaux faibles).

   Il a en outre permis d’en constater les limites : les données, aussi fines soient-elles, ne permettent pas de se placer au niveau des individus, de saisir la complexité des situations individuelles et des sentiments. Pour cela, des entretiens qualitatifs sont nécessaires.
    '''

    st.markdown(multi)



