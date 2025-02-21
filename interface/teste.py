import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

# Carregar tokenizer
TOKENIZER = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Definir classe do modelo
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
        self.fc = torch.nn.Linear(768, 9)  # 9 emoções

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(pooled_output)


# Carregar o melhor modelo
def load_best_model():
    model = BERTClass()  # Instanciar o modelo aqui
    model_path = "C:/Users/ramon/OneDrive/Documentos/GitHub/EmotionsMiningPTBR/interface/model/best_model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Ajusta as chaves para remover "bert_model." do início
    new_state_dict = {k.replace("bert_model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(new_state_dict, strict=False)  # strict=False permite carregar apenas as chaves que coincidem
    model.eval()
    return model


# Função para prever a emoção de uma frase
def predict_emotion(text, model):
    # Tokenização da frase sem expor input_ids e attention_mask
    inputs = TOKENIZER(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    # Previsão diretamente com a entrada tokenizada
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])  # Isso ainda é necessário para BERT
    
    # Obter a classe prevista
    predicted_class = torch.argmax(outputs).item()
    emotion_labels = ['alegria', 'tristeza', 'raiva', 'medo', 'nojo', 'surpresa', 'confianca', 'antecipacao', 'neutro']
    # Retorna a emoção detectada
    return emotion_labels[predicted_class]

# Exemplo de uso
if __name__ == "__main__":
    model = load_best_model()
    while True:
        frase = input("Digite uma frase: ")  # O usuário digita uma frase sem necessidade de tokenização
        if frase == "sair":
            break
        emocao = predict_emotion(frase, model)
        print(f"A emoção detectada é: {emocao}")


