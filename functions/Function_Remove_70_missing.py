# Importo librerie utili
import pandas as pd

# Funzione per rimuovere le colonne e le righe con più del 70% di valori mancanti
# Tali colonne potenzialmente non hanno nessuna informazione utile
# In altri casi, tali colonne possono essere trasformate per conservare le poche informazioni (NON INCLUSO IN QUESTO SCRIPT)
def remove_missing(dataframe, threshold = 0.7) :
    """
    La funzione accetta come unico parametro il dataframe da controllare
    La funzione restituisce il dataframe ripulito da righe e colonne con più del 70% di valori mancanti, e la lista di colonne rimaste
    """
    # Righe e colonne
    Rows = dataframe.shape[0]
    Columns = dataframe.shape[1]

    # Rimozione delle colonne con più del 70% di valori mancanti
    removed_columns = [col for col in dataframe.columns if dataframe[col].isna().mean() > threshold]
    dataframe.drop(columns = removed_columns, inplace = True)

    # Rimozione delle righe con più del 70% di valori mancanti
    row_threshold = int( dataframe.shape[1] * (1 - threshold) )
    dataframe.dropna(thresh = row_threshold, inplace = True)

    
    # Rimozione delle colonne con più del 70% di valori mancanti    
    #Removed_columns = []
    #for i in dataframe.columns :
    #    if dataframe[i].isna().sum()/Rows > 0.7 :
    #        Removed_columns.append(i)
    #        dataframe.drop([i], axis = 1, inplace = True)

    # Rimozione delle righe con più del 70% di valori mancanti
    # Ogni riga che rappresenta un record, se ha pochi valori non ha nessuna informazione utile
    #dataframe.dropna(thresh = int(Columns*0.7), inplace = True)
    #dataframe = dataframe.reset_index()
    #dataframe.drop("index", axis = 1, inplace = True)
    #Selected_columns_start = dataframe.columns

    # Rest index + colonne finali
    dataframe.reset_index(drop = True, inplace = True)
    selected_columns = dataframe.columns
    
    return dataframe, selected_columns
