# ReconhecimentoFacial2
PROJETO RECONHECIMENTO FACIAL

Projeto para reconhecimento facial para destrancar portas ou permitir acesso a pessoas previamente cadastradas

DEPLOYMENT: --------------------------------------------------------------------------------------------
*Seus arquivos devem estar desta forma para o funcionamento dos códigos:*

/Diretorio_Ambiente
|-- /Ambiente
|-- /dataset
*pastas nomeadas com o nome da pessoa e dentro delas fotos desta pessoa*
|----   /fulano
|------     *fotos do fulano*
|-- /face_detection_model
|----   deploy.prototxt
|----   res10_300x300_ssd_iter_140000.caffemodel
|-- /output
|----   embeddings.pickle
|----   le.pickle
|----   recognizer.pickle
|-- /__pycache__
|-- /videos
*videos nomeados com o nome do fulano a ser cadastrada*
|-- codigo_foto_por_video.py
|-- extract_embeddings.py
|-- train_model.py
|-- recognize_video.py

PREREQUISITES: ---------------------------------------------------------------------------------
Necessário a criação de um ambiente (enviroment) usando Virtualenv:
https://www.liquidweb.com/kb/creating-virtual-environment-ubuntu-16-04/

Bibliotecas:
-open_cv2;
-imutils;
-sklearn;
-nativas: numpy, os, pickle, argparse;

----------- melhorar README: https://gist.github.com/PurpleBooth/109311bb0361f32d87a2 ----------

RUNNING: ---------------------------------------------------------------------------------------
*Passos 1.0, 2.0 e 3.0 tratam-se do cadastro da pessoa como reconhecível pelo programa, envolvendo da*
*utilização das fotos de rosto para criar vetores/matrizes e detectar uma pessoa e criar o embeddings que*
*será utlizado pelo código train_model para treinar a rede neural*
*Passo 4.0 é o código de reconhecimento facial que será o código principal (main code)*

1.0) Run o código "codigo_foto_por_video.py" (F5 no vscode/executando pelo terminal)
     1.1) Caso tenha um vídeo já gravado do seu rosto, coloque na pasta /Diretorio_Ambiente/videos com o
     nome do indivíduo a ser cadastrado
2.0) Run o código "extract_embeddings.py" usando o terminal do vscode coloque:
"python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7"
3.0) Run o código "train_model.py" usnado o terminal do vscode coloque:
"python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle"
4.0) Run o código "recognize_video.py" para detectar e reconhecer as faces.

