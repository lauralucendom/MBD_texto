from numpy import array, argmax, random, take
import numpy as np
import string
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class UTILS:
    def __init__(self):
        """
        Constructor de la clase.
        """
        print("Instancia de la clase UTILS creada")

    def read_data(self, filename: str = 'deu-eng/deu_modif.txt'):
        """ 
        Función para leer el texto que se encuentra dentro del archivo especificado y dividir el texto obtenido en pares alemán/inglés o inglés/español.

        Args:
            filename (str): archivo del que se va a leer el texto.
        """

        file = open(filename, mode='rt', encoding='utf-8')
        text = file.read()
        file.close()

        sents = text.strip().split('\n')
        sents = [i.split('\t') for i in sents]
        sents = array(sents)
    
        return sents

    def text_preprocessing(self, data: np.array):
        """
        Función para eliminar los signos de puntuación de cada frase y para pasar a minúscula cada una de ellas.
        
        Args: 
            data (np.array) = np.array con los pares alemán/inglés o inglés/español a preprocesar.
        """
        
        data[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:,0]]
        data[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in data[:,1]]

        for i in range(len(data)):
            data[i,0] = data[i,0].lower()
            data[i,1] = data[i,1].lower()
        
        return data

    def get_length(self, data: np.array):
        """
        Función para calcular la longiutd de cada una de las oraciones de los pares existentes. 
        En este caso se usan pares inglés/español. En caso de querer cambiarlo, habría que cambiar los nombres de las columnas del dataframe obtenido.
        
        Args: 
            data (np.array) = np.array con los pares alemán/inglés o inglés/español ya preprocesados.
        """

        l1 = []
        l2 = []

        for i in data[:,0]:
            l1.append(len(i.split()))

        for i in data[:,1]:
            l2.append(len(i.split()))

        length_df = pd.DataFrame({'eng':l1, 'esp':l2})
    
        return max(l1),max(l2),length_df
    
    def tokenization(self, columna):
        """
        Función para construir el tokenizer.
        
        Args: 
            columna = columna del df sobre la que queremos aplicar el tokenizer.
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(columna)
        
        return tokenizer.texts_to_sequences(columna), tokenizer

    def encode_sequences(self, tokenizer, length, columna):
        """
        Función para preparar las secuencias. También realizará el relleno de secuencia hasta una longitud máxima de oración que viene asignada por el argumento length.
        
        Args: 
            tokenizer = se especifica el tokenizer a usar.
            length = longitud máxima de oración.
            columna = columna en la que se quiere aplicar la codificación y creación de secuencias.
        """

        seq = tokenizer.texts_to_sequences(columna)

        seq = pad_sequences(seq, maxlen=length, padding='post')

        print(seq)
        print(len(seq))

        return seq

    def get_word(self, n, tokenizer):
        """
        Función para convertir las secuencias predichas en palabras a partir de un tokenizer y un índice.
        
        Args: 
            n = número que debe matchear con el índice
            tokenizer = se especifica el tokenizer a usar.

        """

        for word, index in tokenizer.word_index.items():
            if index == n:
                return word
        return None

class PLOTS:
    def __init__(self):
        """
        Constructor de la clase.
        """
        print("Instancia de la clase PLOTS creada")

    def plot_history(self, history): 

        """
        Plot Keras History
        Plot loss and accuracy for the training and validation set.
        Args:
            history ([type]): history.
        """ 

        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
        plt.figure(figsize=(22,10))
        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
        ## Accuracy
        plt.figure(221, figsize=(20,10))
        ## Accuracy
        # plt.figure(2,figsize=(14,5))
        plt.subplot(221, title='Accuracy')
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        ## Loss
        plt.subplot(222, title='Loss')
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))    
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
