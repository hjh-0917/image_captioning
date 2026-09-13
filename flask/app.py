from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import io
import os 

version = 5
DIRPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_PATH = os.path.join(f'{DIRPATH}','input/Vocabulary.json')
            
WEIGHT_PATH = os.path.join(f'{DIRPATH}', f'weight/latest.pth')

with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

idx_to_token = list(vocab.keys())
word_count = len(idx_to_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


def make_cnn_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )


class algorithem1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = nn.Sequential(
            make_cnn_block(3,   64),
            make_cnn_block(64,  128),
            make_cnn_block(128, 256),
            make_cnn_block(256, 512),
            make_cnn_block(512, 512),
            nn.AdaptiveAvgPool2d((14, 14))
        )
        self.image_to_features = nn.Linear(512, 768)
        self.word_embedding = nn.Embedding(word_count, 768)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8),
            num_layers=4
        )
        self.llm_model = nn.Sequential(nn.Linear(768, word_count))

    def forward(self, img_feat, previous_embedding_ids):
        word_feat = self.word_embedding(previous_embedding_ids).unsqueeze(1)
        tgt_len = word_feat.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
        output = self.decoder(tgt=word_feat, memory=img_feat, tgt_mask=tgt_mask)
        return self.llm_model(output[-1])


model = algorithem1().to(device)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
])


def generate_caption(image, max_len=30):
    img = image.convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = model.cnn_model(img)
        img_feat = img_feat.flatten(2).permute(2, 0, 1)
        img_feat = model.image_to_features(img_feat)

        current_context = [0]

        for _ in range(max_len):
            input_tensor = torch.tensor(current_context, dtype=torch.long).to(device)
            output = model(img_feat, input_tensor)
            next_token = output.argmax().item()

            if idx_to_token[next_token] == 'SEP':
                break

            current_context.append(next_token)

    tokens = [idx_to_token[i] for i in current_context[1:]]
    caption = ' '.join(tokens)
    caption = caption.replace(' ##', '')
    return caption


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        image = Image.open(io.BytesIO(image.read()))
        caption = generate_caption(image)
        return jsonify({'caption': caption})
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)