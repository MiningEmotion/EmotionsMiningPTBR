import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import os

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
    model = BERTClass()
    model_path = "C:/Users/ramon/OneDrive/Documentos/GitHub/EmotionsMiningPTBR/interface/model/best_model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {k.replace("bert_model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

# Função para prever a emoção de uma frase
def predict_emotion(text, model):
    inputs = TOKENIZER(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    probabilities = F.softmax(outputs, dim=1).squeeze()
    predicted_class = torch.argmax(probabilities).item()
    emotion_labels = ['alegria', 'tristeza', 'raiva', 'medo', 'nojo', 'surpresa', 'confianca', 'antecipacao', 'neutro']
    return emotion_labels[predicted_class]

# Função para atualizar a imagem com base na emoção
def update_image(emotion):
    image_path = f"C:/Users/ramon/OneDrive/Documentos/GitHub/EmotionsMiningPTBR/interface/roda das emocoes/{emotion}.png"
    if os.path.exists(image_path):
        img = Image.open(image_path)
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        img_width, img_height = img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        canvas.itemconfig(emotion_image_id, image=img)
        canvas.coords(emotion_image_id, canvas_width // 2, canvas_height // 2)
        canvas.image = img
    else:
        canvas.itemconfig(emotion_image_id, image='')

# Função para redimensionar a imagem de fundo
def resize_background(event):
    if os.path.exists(background_image_path):
        background_img = Image.open(background_image_path)
        img_width, img_height = background_img.size
        canvas_width = event.width
        canvas_height = event.height
        if canvas_width > 0 and canvas_height > 0:
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            background_img = background_img.resize(new_size, Image.LANCZOS)
            background_img = ImageTk.PhotoImage(background_img)
            canvas.itemconfig(background_image_id, image=background_img)
            canvas.coords(background_image_id, canvas_width // 2, canvas_height // 2)
            canvas.background_img = background_img

# Função para processar a entrada do usuário
def process_input():
    frase = entry.get()
    if frase:
        emocao = predict_emotion(frase, model)
        result_label.config(text=f"A emoção detectada é: {emocao}")
        update_image(emocao)

# Carregar o modelo
model = load_best_model()

# Configurar a interface gráfica
root = tk.Tk()
root.title("Detecção de Emoções")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(2, weight=1)

entry_label = ttk.Label(frame, text="Digite uma frase:")
entry_label.grid(row=0, column=0, padx=5, pady=5)

entry = ttk.Entry(frame, width=50)
entry.grid(row=0, column=1, padx=5, pady=5)

button = ttk.Button(frame, text="Detectar Emoção", command=process_input)
button.grid(row=0, column=2, padx=5, pady=5)

result_label = ttk.Label(frame, text="")
result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

# Configurar o Canvas para exibir as imagens
canvas = tk.Canvas(frame)
canvas.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

# Carregar e exibir a imagem de fundo
background_image_path = "C:/Users/ramon/OneDrive/Documentos/GitHub/EmotionsMiningPTBR/interface/roda das emocoes/roda-plutchik.png"
background_image_id = canvas.create_image(0, 0, anchor=tk.CENTER)
emotion_image_id = canvas.create_image(0, 0, anchor=tk.CENTER)

# Bind para redimensionar a imagem de fundo
canvas.bind('<Configure>', resize_background)

root.mainloop()