from os import scandir, getcwd
from os.path import abspath
from collections import Counter 
import re
import numpy as np
class SpamDetector:
  import numpy as np
  def __init__(self):
    
    """El método init leerá todos los archivos con extensión ".txt" de los directorios datasets/train y datasets/test y los usará para crear las matrices de X_train, X_test, Y_train y Y_test.

    Cada archivo ".txt" corresponderá a una instancia de entrenamiento o prueba. El formato de los archivos ".txt" es el siguiente:

    <0|1>
    <contenido>

    La primera línea contendrá la etiqueta 0 si el contenido no es spam y 1 si lo es. A partir de la segunda línea todo el texto que se encuentre se considerará parte del correo electrónico.

    El texto de un correo electrónico se convertirá en un vector de atributos de la siguiente
    forma: 

    Primero se detectarán todas las palabras distintas (sólo letras, ningún símbolo) que
    aparezcan todos los archivos (train + test), luego a cada palabra se le asignará un índice y
    finalmente se representará cada correo a través de un vector cuyos elementos reprsentarán la
    cantidad de veces que aparece cada palabra en su contenido de acuerdo al índice que se asignó.

    Ejemplo:

    Si tuviésemos solamente tres palabras distintas: no, si, spam, le asignaríamos a cada un índice de 0 al 2 (esta longitud representa a n). Y si el contenido de un correo fuese: "spam spam si si" el vector que lo representaría sería: 
    [0 2 2]

    Recuerde que todos los vectores deben tener la misma longitud. 

    Las URL se considerarán como palabras, ej.: http://www.google.com sería cuatro palabras: http, www, google y com.

    PD: No olvide que es recomendable estandarizar los datos de entrenamiento y de prueba.
    
    """
   
    #Y_test  e Sacando las palabras de todos los X test
    self.ruta=r'datasets/test'
    self.rutaAbsoluta=[abspath(arch.path) for arch in scandir(self.ruta) if arch.is_file()]
    self.yT=[]
    
    self.xTest=[]#Guardara X_test
    self.lista=[]
    self.f=[]
    self.listOfList=[]
    for i in range(len(self.rutaAbsoluta)): 
       #print(self.rutaAbsoluta[i])
       self.fopen=open(self.rutaAbsoluta[i],"r")
       self.lineas=self.fopen.read()
       self.yT.append(float(self.lineas[0])) # vector Y_Test
     
       #X_test  cadena y lista
       #print(self.lineas)# imprime todos los txt con simbolos y numeros
       #for e in range(len(self.lineas)):
       self.cadenaMinusculas = self.lineas.lower().replace("."," ").replace("¿"," ").replace("?"," ").replace("¡"," ").replace("!"," ").replace(","," ").replace(")"," ").replace("("," ").replace(":"," ").replace("="," ").replace("@"," ").replace("/"," ").replace("<"," ").replace(">"," ").replace("$"," ").replace("#"," ").replace("\\"," ").replace("*"," ").replace("-"," ").replace("_"," ").replace("--"," ").replace("š"," ").replace(";"," ").replace("\\\\"," ").replace("”"," ").replace("–"," ").replace("“"," ").replace("รก"," ").replace("©"," ").replace("..."," ").replace("·"," ").replace("|"," ").replace("%"," ").replace("'"," ").replace('"'," ").replace("["," ").replace("]"," ").replace("ˇ"," ").replace("————————————————————————————"," ").replace("+"," ").replace("đ"," ").replace("♈̷̴"," ").replace("贸"," ").replace("_______________"," ").replace("รณ"," ")     
       self.s=re.sub(r'[^\w]', ' ', self.cadenaMinusculas)                     
       self.quitaNumeros = re.sub(r'\d+', '', self.s)
       #print(self.quitaNumeros)#en este punto imprime cada cadena de acuerdo a cada iteracion sin los caracteres de antes en minuscula y sin numeros, si imprimo esto fuera del for solo traeria la ultima cadena por lo tando se creo la candena vacio para poder concatenar todas las cadena e imprimirlas todas desde fuera del for
       self.lista = self.quitaNumeros.split() 
       self.listOfList.append(self.lista)
       self.f= self.f+self.lista
       self.fopen.close()
       
       #print('\n')
    #print(self.vacio)#imprime todas las palabras de los txt de la carpeta test en una sola cadena   
    #print(self.f) #imprime todas las palabras de los txt de la carpeta test en una sola lista
    #print('\n')
    #print('\n')
    #print(self.listOfList)# imprime la lista de listas de las palabras de cada txt ejemplo txt1= hola soy  txt2=si esta   imprime [ [hola, soy] , [si,esta] ]    
    self.n=len(self.f)# tamanio de lista de palabras de carpeta train
    #print(self.yT) 
    

    
    
    #Y_train e Sacando las palabras de todos los Xtrain
    self.ruta1=r'datasets/train'
    self.rutaAbsoluta1=[abspath(arch1.path) for arch1 in scandir(self.ruta1) if arch1.is_file()]
    self.yT1=[] 
    
    self.xTrain1=[]#guardara X_Train
    self.lista1=[]
    self.f1=[]
    self.listOfList1=[]
    for j in range(len(self.rutaAbsoluta1)): 
       #print(self.rutaAbsoluta1[j])#imprime las  rutas absolutas del vector self.rutaAbsoluta
       self.fopen1=open(self.rutaAbsoluta1[j],"r")
       self.lineas1=self.fopen1.read()
       self.yT1.append(float(self.lineas1[0])) #vector Y_Train
   
      
       #X_train  cadena y lista
       #print(self.lineas1)# imprime todos los txt con simbolos y numeros
       #for e in range(len(self.lineas1)):
       self.cadenaMinusculas1 = self.lineas1.lower().replace("."," ").replace("¿"," ").replace("?"," ").replace("¡"," ").replace("!"," ").replace(","," ").replace(")"," ").replace("("," ").replace(":"," ").replace("="," ").replace("@"," ").replace("/"," ").replace("<"," ").replace(">"," ").replace("$"," ").replace("#"," ").replace("\\"," ").replace("*"," ").replace("-"," ").replace("_"," ").replace("--"," ").replace("š"," ").replace(";"," ").replace("\\\\"," ").replace("”"," ").replace("–"," ").replace("“"," ").replace("รก"," ").replace("©"," ").replace("..."," ").replace("·"," ").replace("|"," ").replace("%"," ").replace("'"," ").replace('"'," ").replace("["," ").replace("]"," ").replace("ˇ"," ").replace("————————————————————————————"," ").replace("+"," ").replace("đ"," ").replace("♈̷̴"," ").replace("贸"," ").replace("_______________"," ").replace("รณ"," ")  
       self.s1=re.sub(r'[^\w]', ' ', self.cadenaMinusculas1)                                      
       self.quitaNumeros1 = re.sub(r'\d+', '', self.s1)
       #print(self.quitaNumeros1)#en este punto imprime cada cadena de acuerdo a cada iteracion sin los caracteres de antes en minuscula y sin numeros, si imprimo esto fuera del for solo traeria la ultima cadena por lo tando se creo la candena vacio1 para poder concatenar todas las cadena e imprimirlas todas desde fuera del for
       self.lista1 = self.quitaNumeros1.split() 
       self.listOfList1.append(self.lista1)
       self.f1= self.f1+self.lista1
       self.fopen1.close()
       
       #print('\n')
    #print(self.vacio1)#imprime todas las palabras de los txt de la carpeta train en una sola cadena   
    #print(self.f1) #imprime todas las palabras de los txt de la carpeta train en una sola lista
    #print('\n')
    #print('\n')
    #print(self.listOfList1)# imprime la lista de listas de las palabras de cada txt ejemplo txt1= hola soy  txt2=si esta   imprime [ [hola, soy] , [si,esta] ]
    #self.n1=len(self.f1)# tamanio de lista de palabras de carpeta train
    #print(self.yT1) 
    
    
    
    
    
    self.totalListas=[]
    self.totalListasPD=[]
    
    
    self.totalListas= self.f+self.f1
    #print(self.totalListas) #imprime una lista de todas  todas las palabras (train+test)
    self.totalListasPD=list(set(self.totalListas))
    #self.t=len(self.totalListas)#imprime tamanio tamanio de la lista de palabras de train + test
    #print(self.totalListasPD)#imprime tamanio de lista de todas  todas las palabras distintas( train+test)
    
    self.VectorTest=[]
    for k in range(len(self.listOfList)):
      self.vectorCorreo= [0]*(len(self.totalListasPD))
      for o in range(len(self.listOfList[k])):
        #if (self.totalListasPD[self.totalListasPD.index(self.listOfList[k][o])] == self.listOfList[k][o]):
         self.vectorCorreo[ self.totalListasPD.index(self.listOfList[k][o]) ]= self.listOfList[k].count(self.listOfList[k][o])
      self.VectorTest.append(self.vectorCorreo)
    #print(self.VectorTest)
    #print(len(self.VectorTest))
    
   
  
    self.VectorTrain=[]
    for k in range(len(self.listOfList1)):
      self.vectorCorreo1= [0]*(len(self.totalListasPD))
      for o in range(len(self.listOfList1[k])):
        #if (self.totalListasPD[self.totalListasPD.index(self.listOfList1[k][o])] == self.listOfList1[k][o]):
         self.vectorCorreo1[ self.totalListasPD.index(self.listOfList1[k][o]) ]= self.listOfList1[k].count(self.listOfList1[k][o])
      self.VectorTrain.append(self.vectorCorreo1)
    #print(self.VectorTrain)
    #print(len(self.VectorTrain))
    
    
    self.TR=self.np.array(self.VectorTrain)
    self.TE=self.np.array(self.VectorTest)
    self.Tra=np.array(self.yT1)
    self.Tes=np.array(self.yT)
    
    self.train_set_X_flatten=self.TR.reshape(self.TR.shape[0],-1).T
    self.test_set_X_flatten=self.TE.reshape(self.TE.shape[0],-1).T
    self.train_set_Y_flatten=self.Tra.reshape(self.Tra.shape[0],-1).T
    self.test_set_Y_flatten=self.Tes.reshape(self.Tes.shape[0],-1).T
    #print(self.train_set_X_flatten.shape)#imprime tamanio de transpuesta de X train
    #print(self.test_set_X_flatten.shape)#imprime tamanio de transpuesta de X test
    #print(self.train_set_Y_flatten.shape)#imprime tamnio de traspuesta de Y train
    #print(self.test_set_Y_flatten.shape)#imprime tamnio de traspuesta de Y test
    
    #print(self.train_set_X_flatten)#imprime la tranpuesta de X train
    #print(self.test_set_X_flatten)#imprime la transpuesta de X test
    #print(self.train_set_Y_flatten)#imprime la transpuesta de Y train
    #print(self.test_set_Y_flatten)#imprime la transpuesta de Y test
   
  
   
   
  
    
    
    #Normalizando train_set_X_flatte e Normalizando test_set_X_flatten con norma de train

    self.d=np.array(self.train_set_X_flatten)
    
    self.dExp=np.exp(self.d)
    self.dSum=np.sum(self.dExp,axis=1,keepdims=True)
    self.r=self.dExp/self.dSum
    

    self.a=np.array(self.test_set_X_flatten)
    
    self.tExp=np.exp(self.r)
    self.tSum=np.sum(self.tExp,axis=1,keepdims=True)
    self.r1=self.tExp/self.tSum
    #print(self.r1)
    
        
                      
  
  def train_model(self, X_train, Y_train, num_iterations, learning_rate):
    """Entrena un modelo de regresión logística con los parámetros recibidos.

    Una vez finalizado el entrenamiento retorna un diccionario con los siguientes datos:
    { "costs": costs,  Una lista con los costos obtenidos cada 100 iteraciones
      "w" : w,         Un numpy array con el valor final del parámetro w.
      "b" : b,         Un número tipo float con el valor final del parámetro b.
    }
    """
 
    self.w=np.zeros((X_train.shape[0],1))
    self.b=0.0
    self.m=X_train.shape[1]
    self.print_cost=False           
    self.costs=[]
    assert(self.w.shape==(X_train.shape[0],1))
    assert(isinstance(self.b,float) or isinstance(self.b,int))   
    for i in range(num_iterations):
       self.Z=np.dot(self.w.T,X_train)+self.b          #self.w.T.dot(X_train)+self.b
       self.Z=np.array(self.Z)
       self.A=1.0/(1.0+np.exp(-self.Z))
       self.cost=(-1.0/self.m)*np.sum(Y_train*np.log(self.A)+(1.0-Y_train)*np.log(1.0-self.A))
       self.dZ=self.A-Y_train
       self.dw=(1.0/self.m)*np.dot(X_train,self.dZ.T)
       self.db=(1.0/self.m)*np.sum(self.dZ)
       assert(self.dw.shape==self.w.shape)
       assert(self.db.dtype==float)
       self.cost=np.squeeze(self.cost)
       assert(self.cost.shape==())
       self.w=self.w-learning_rate * self.dw
       self.b=self.b-learning_rate * self.db
       if i % 100==0:
          self.costs.append(self.cost)
          
       #print(self.cost)
       #print(self.w)
       #print(self.b)
       #print('/n')
       
       
    #print(self.cost)
    #print(self.w[num_iterations])
    #print(self.w.shape)
    #print(self.b)  
    self.Y_prediction_train=np.zeros((1,self.m))
    self.w=self.w.reshape(X_train.shape[0],1)
    self.A=1.0/(1.0+np.exp(-(self.w.T.dot(X_train)+self.b)))
    self.n=self.A.shape[1]
    for i in range(self.n):
        self.Y_prediction_train[0,i]= 1.0 if self.A [0,i] >= 0.5 else 0.0
    assert(self.Y_prediction_train.shape==(1,self.m))
    #print(self.Y_prediction_train)
    print("train accuracy: {} %".format(100-np.mean(np.abs(self.Y_prediction_train-Y_train))*100))
    
    
    
    self.Y_prediction_test=np.zeros((1,self.m))
    self.w=self.w.reshape(self.r1.shape[0],1)
    self.A=1.0/(1.0+np.exp(-(self.w.T.dot(self.r1)+self.b)))
    self.n=self.A.shape[1]
    for i in range(self.n):
        self.Y_prediction_test[0,i]= 1.0 if self.A [0,i] >= 0.5 else 0.0
    assert(self.Y_prediction_test.shape==(1,self.m))
    #print(self.Y_prediction_train)
    print("test accuracy: {} %".format(100-np.mean(np.abs(self.Y_prediction_test-self.r1))*100))
    
    self.dicDataModel={}
    self.dicDataModel['costs']=self.costs
    self.dicDataModel['w']=self.w
    self.dicDataModel['b']=self.b
    return self.dicDataModel
    

    
    
    
    
    
    
  

  def get_datasets(self):
    """Retorna un diccionario con los datasets preprocesados por el método init estos mismos son
    los que deben ser usados para el entrenamiento.
    
    { "X_train": X_train,
      "X_test": X_test,
      "Y_train": Y_train,
      "Y_test": Y_test
    }
    
      
      return { "X_train": np.zeros((8000,280)),
      "X_test": np.zeros((8000,119)),
      "Y_train": np.zeros((1,280)),
      "Y_test": np.zeros((1,119)) }
      
      

    """
    self.dicData={}
    self.dicData['X_train']= self.train_set_X_Normalizado
    self.dicData['X_test']=self.test_set_X_Normalizado
    self.dicData['Y_train']=self.train_set_Y_flatten
    self.dicData['Y_test']=self.test_set_Y_flatten
    
    return self.dicData

  def get_best_training_config(self):
    """Retorna un diccionario con los datos que el estudiante encontró son los mejores para 
    realizar el entrenamiento y obtener la precisión mínima requerida, usando los datasets 
    obtenidos por get_datasets
    
    { "w": w,                          Valor del parámetro w (numpy array)
      "b": b,                          Valor del parámetro b.
      "learning_rate": learning_rate   Valor del ritmo de aprendizaje.
      "num_iterations": num_iterations Número de iteraciones realizadas.
      "train_accuracy": train_accuracy Precisión del modelo en X_train de 0 a 100.
      "test_accuracy": test_accuracy   Precisión del modelo en X_test de 0 a 100.
    }
    """
    self.dicBest={}
    self.dicBest['w']= [0.0018497]
    self.dicBest['b']=-0.08997894126461103
    self.dicBest['learning_rate']='0.1'
    self.dicBest['num_iterations']='2000'
    self.dicBest['train_accuracy']='63.92857142857143'
    self.dicBest['test_accuracy']='99.64285714285714'
    
    return self.dicBest
  
    

  
  
SD=SpamDetector()
SD.train_model(SD.r,SD.train_set_Y_flatten,2000,0.1)
#SD.train_model(np.zeros((8780,280)),np.zeros((1,280)),2000,0.5)


