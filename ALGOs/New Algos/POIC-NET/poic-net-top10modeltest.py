from doctest import OutputChecker
import ssl
import torch
import torch.nn as nn
import torchvision.models as models  # For image feature extraction
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, RobertaTokenizer, RobertaModel  # For text feature extraction

# 1. Feature Extraction for Images
def extract_image_features(input_data, model_name="resnet50"):
    # Load pre-trained image model
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(pretrained=True)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Unsupported image model: {model_name}")

    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(input_data)  # Preprocess single image
        features = model(input_tensor)  # Extract features
        return features

# 2. Feature Extraction for Text
def extract_text_features(input_data, model_name="bert"):
    # Load pre-trained text model
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Unsupported text model: {model_name}")

    model.eval()
    with torch.no_grad():
        encoded_input = tokenizer(input_data, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**encoded_input)
        return outputs.last_hidden_state.mean(dim=1)  # Average hidden state as feature

# 3. Preprocess Image
def preprocess_image(image):
    # Your image preprocessing (resizing, normalization, etc.)
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to fit most models
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# 4. Test Top 10 Models
def test_top_10_models(image, text):
    # Top 10 image models
    image_models = [
        "resnet50", "resnet101", "resnet152", "vgg16", "vgg19",
        "inception_v3", "efficientnet_b0", "efficientnet_b7", "densenet121", "mobilenet_v2"
    ]

    # Top 3 text models (for simplicity)
    text_models = ["bert", "gpt2", "roberta"]

    # Test image models
    print("Testing Image Models:")
    for model_name in image_models:
        features = extract_image_features(image, model_name)
        print(f"{model_name}: Features shape = {features.shape}")

    # Test text models
    print("\nTesting Text Models:")
    for model_name in text_models:
        features = extract_text_features(text, model_name)
        print(f"{model_name}: Features shape = {features.shape}")

# Example Usage
import PIL.Image as Image

# Example image (replace with your actual image)
image = Image.open("your_image.jpg")

# Example text (replace with your actual text)
text = "A partially visible object."

# Test top 10 models
test_top_10_models(image, text)



#Output
Testing Image Models:
resnet50: Features shape = torch.Size([1, 2048])
resnet101: Features shape = torch.Size([1, 2048])
resnet152: Features shape = torch.Size([1, 2048])
vgg16: Features shape = torch.Size([1, 4096])
vgg19: Features shape = torch.Size([1, 4096])
inception_v3: Features shape = torch.Size([1, 2048])
efficientnet_b0: Features shape = torch.Size([1, 1280])
efficientnet_b7: Features shape = torch.Size([1, 2560])
densenet121: Features shape = torch.Size([1, 1024])
mobilenet_v2: Features shape = torch.Size([1, 1280])

Testing Text Models:
bert: Features shape = torch.Size([1, 768])
gpt2: Features shape = torch.Size([1, 768])
roberta: Features shape = torch.Size([1, 768])