# Importo librerie utili
import pandas as pd

# Funzione per rimuovere le colonne e le righe con più del 70% di valori mancanti
# Tali colonne potenzialmente non hanno nessuna informazione utile
# In altri casi, tali colonne possono essere trasformate per conservare le poche informazioni (NON INCLUSO IN QUESTO SCRIPT)
def Remove_70_missing(dataframe) :
    """
    La funzione accetta come unico parametro il dataframe da controllare
    La funzione restituisce il dataframe ripulito da righe e colonne con più del 70% di valori mancanti, e la lista di colonne rimaste
    """
    # Righe e colonne
    Rows = dataframe.shape[0]
    Columns = dataframe.shape[1]

    # Rimozione delle colonne con più del 70% di valori mancanti    
    Removed_columns = []
    for i in dataframe.columns :
        if dataframe[i].isna().sum()/Rows > 0.7 :
            Removed_columns.append(i)
            dataframe.drop([i],axis=1,inplace=True)

    # Rimozione delle righe con più del 70% di valori mancanti
    # Ogni riga che rappresenta un record, se ha pochi valori non ha nessuna informazione utile
    dataframe.dropna(thresh = int(Columns*0.7), inplace=True)
    dataframe = dataframe.reset_index()
    dataframe.drop("index", axis=1, inplace=True)
    Selected_columns_start = dataframe.columns

    return dataframe, Selected_columns_start
