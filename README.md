<div align="left">
  <a href="https://github.com/MiningEmotion/EmotionsMiningPTBR">
  
  ![EmotionMining](https://github.com/MiningEmotion/EmotionsMiningPTBR/assets/171974518/05b49e79-8ca9-4bef-ba26-d633e9719299)
  
  <a/>
  <h1>EmotionsMiningPTBR</h1>

<ul>
  <li><a href="#introducao">Introdução</a></li>
  <li><a href="#conteudos">Conteúdos</a></li>
  <ul>
    <li><a href="#tweetemotionptbr">tweetEmotionPTBR.xlsx</a></li>
    <li><a href="#tweetsprocessado">Tweets-processado.xlsx</a></li>
    <li><a href="#fraseschatgpt">frasesChatgpt.xlsx</a></li>
    <li><a href="#emotionminingsvm">EmotionMiningPTBR_SVM.ipynb</a></li>
    <li><a href="#emotionminingsvmbert">bert_multilabel_pytorch_standard.ipynb</a></li>
  </ul>
  <li><a href="#resultados">Resultados</a></li>
  <li><a href="#requirements">Requirements</a></li>

</ul>

<h1></h1>

<h2><a name="introducao">&#x1F4D6 Introdução</a></h2>

<p>➥ Este repositório apresenta scripts de um projeto de pesquisa sobre uma abordagem multirrótulo para a detecção de emoções expressas em textos curtos em português brasileiro. Para isso, propõe-se o emprego da roda de emoções de Plutchik como ferramenta teórica norteadora à construção de um corpus rotulado com tweets coletados por Web Scraper. Além disso, são apresentadas as etapas realizadas no pré-processamento do corpus e no treinamento de modelos de aprendizado de máquina, SVM e BERT, para a classificação emocional de textos gerados por uma LLM.</p>


<h1></h1>

<h2><a name="conteudos">👨‍💻 Conteúdos</a></h2>
<ul type="none">
  <li><h3><a name="tweetemotionptbr">	🗂️ tweetEmotionPTBR.xlsx</a></h3></li>
  
<p>➥ Corpus construído a partir de publicações na rede social X, obtidos por um Web Scraper programado em Python com a biblioteca Selenium. A coleta dos textos foi realizada por meio da busca de sinônimos das emoções primárias e secundárias. As emoções secundárias foram identificadas como aquelas que caracterizaram o conjunto de dados como multirrótulo, com base na roda de emoções teorizada por Plutchik.</p>

  <li><h3><a name="tweetsprocessado"> 📁 Tweets-processado.xlsx</a></h3></li>
  

<p>➥ Corpus tweetEmotionPTBR.xlsx pré-processado para o treino e teste do modelo de aprendizagem de máquina profundo BERT (Bidirectional Encoder Representations from Transformers).</p>


  <li><h3><a name="fraseschatgpt"> 🤖 frasesChatgpt.xlsx</a></h3></li>
  

<p>➥ Dados não vistos gerados com a Large Language Model (LLM) de Inteligência Artificial (IA) ChatGPT 3.5 para testar os modelos de aprendizado de máquina. Foram geradas 10 frases para cada emoção secundária, totalizando 80 frases que tiveram por objetivo simular textos curtos que expressem as duas emoções primárias que compõem a secundária.</p>

  <li><h3><a name="emotionminingsvm">💻 EmotionMiningPTBR_SVM.ipynb</a></h3></li>
  
  Script de treino e teste do modelo máquinas de vetores de suporte (Support Vector Machine - SVM). Possiu as importações, leitura e pré-processamento do corpus, treino, teste e os resultados do desempenho do modelo na detecção de múltiplas emoções em textos curtos.

  <li><h3><a name="emotionminingsvmbert">🖥️bert_multilabel_pytorch_standard.ipynb</a></h3></li>
  
  Script de treino e teste do modelo BERT. Possiu as importações, leitura, treino, teste e os resultados do desempenho do modelo na detecção de múltiplas emoções em textos curtos.

</ul>

<h1></h1>

<h2><a name="resultados">📊 Resultados</a></h2>
  
![ComparacaoSVM_BERT](https://github.com/MiningEmotion/EmotionsMiningPTBR/assets/171974518/cbc3c4bf-61b5-4ae4-85d2-b44b9af173ca)

![MétricaPorEmoção_BERT_SVM](https://github.com/MiningEmotion/EmotionsMiningPTBR/assets/171974518/02839773-021e-42a1-aa0b-1fb2916cbd7b)

<h1></h1>

<h2><a name="requirements">&#x2699 Requirements</a></h2>

<p>➥ Este projeto requer as seguintes bibliotecas e ferramentas para execução adequada: </p>

~~~Python

joblib==1.1.0
matplotlib==3.7.1
nltk==3.7.1
numpy==1.22.0
pandas==1.4.0
scikit-learn==1.1.0
scikit-multilearn==0.2.0
seaborn==0.12.2
torch==1.10.0
transformers==4.12.2

~~~

<p>➥ Para instalar todas as dependências necessárias, utilize o comando: 

~~~
pip install -r requirements.txt
~~~
