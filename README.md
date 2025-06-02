# Computer Vision Object Detection

[English](#english) | [Português](#português)

## English

### Overview
Advanced computer vision system for real-time object detection using OpenCV and deep learning models. Features web interface, multiple detection algorithms, and professional visualization capabilities for identifying and tracking objects in images and video streams.

### Features
- **Real-Time Detection**: Live object detection from webcam or video files
- **Multiple Models**: Support for YOLO, SSD, and custom trained models
- **Web Interface**: Professional Flask application with file upload
- **Batch Processing**: Analyze multiple images simultaneously
- **Confidence Filtering**: Adjustable confidence thresholds
- **Bounding Boxes**: Visual object localization with labels
- **Performance Metrics**: Detection accuracy and processing speed
- **Export Results**: Save annotated images and detection reports

### Technologies Used
- **Python 3.8+**
- **OpenCV**: Computer vision library
- **Flask**: Web framework
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Deep Learning**: Pre-trained neural networks

### Installation

1. Clone the repository:
```bash
git clone https://github.com/galafis/Computer-Vision-Object-Detection.git
cd Computer-Vision-Object-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python object_detector.py
```

4. Open your browser to `http://localhost:5000`

### Usage

#### Web Interface
1. **Upload Image**: Select image file for object detection
2. **Adjust Settings**: Set confidence threshold and model parameters
3. **Process**: Click detect to analyze the image
4. **View Results**: See detected objects with bounding boxes and labels
5. **Download**: Save annotated image with detection results

#### API Endpoints

**Image Detection**
```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@your_image.jpg" \
  -F "confidence=0.5"
```

**Batch Processing**
```bash
curl -X POST http://localhost:5000/api/batch_detect \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
```

#### Python API
```python
from object_detector import ObjectDetector

detector = ObjectDetector()

# Load and process image
image_path = "sample_image.jpg"
detections = detector.detect_objects(image_path)

# Display results
for detection in detections:
    print(f"Object: {detection['class']}")
    print(f"Confidence: {detection['confidence']:.2f}")
    print(f"Bounding Box: {detection['bbox']}")
```

### Supported Objects
The system can detect 80+ object classes including:
- **People**: Person detection and tracking
- **Vehicles**: Cars, trucks, motorcycles, bicycles
- **Animals**: Dogs, cats, birds, horses
- **Objects**: Chairs, tables, laptops, phones
- **Food**: Fruits, vegetables, drinks
- **Sports**: Balls, equipment, activities

### Detection Models

#### YOLO (You Only Look Once)
- Fast real-time detection
- Single neural network evaluation
- Good balance of speed and accuracy

#### SSD (Single Shot Detector)
- Multi-scale feature detection
- Efficient for mobile deployment
- Optimized for speed

#### Custom Models
- Train with your own datasets
- Fine-tune for specific use cases
- Domain-specific object detection

### Performance Optimization
- **GPU Acceleration**: CUDA support for faster processing
- **Model Quantization**: Reduced model size for mobile deployment
- **Batch Processing**: Efficient multi-image analysis
- **Caching**: Smart caching for repeated detections

### Configuration
Customize detection parameters in `config.json`:
```json
{
  "confidence_threshold": 0.5,
  "nms_threshold": 0.4,
  "input_size": 416,
  "model_path": "models/yolo.weights"
}
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Português

### Visão Geral
Sistema avançado de visão computacional para detecção de objetos em tempo real usando OpenCV e modelos de deep learning. Apresenta interface web, múltiplos algoritmos de detecção e capacidades de visualização profissional para identificar e rastrear objetos em imagens e streams de vídeo.

### Funcionalidades
- **Detecção em Tempo Real**: Detecção ao vivo de objetos via webcam ou arquivos de vídeo
- **Múltiplos Modelos**: Suporte para YOLO, SSD e modelos treinados personalizados
- **Interface Web**: Aplicação Flask profissional com upload de arquivos
- **Processamento em Lote**: Analise múltiplas imagens simultaneamente
- **Filtragem de Confiança**: Limites de confiança ajustáveis
- **Caixas Delimitadoras**: Localização visual de objetos com rótulos
- **Métricas de Performance**: Precisão de detecção e velocidade de processamento
- **Exportar Resultados**: Salve imagens anotadas e relatórios de detecção

### Tecnologias Utilizadas
- **Python 3.8+**
- **OpenCV**: Biblioteca de visão computacional
- **Flask**: Framework web
- **NumPy**: Computação numérica
- **Pillow**: Processamento de imagens
- **Deep Learning**: Redes neurais pré-treinadas

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/galafis/Computer-Vision-Object-Detection.git
cd Computer-Vision-Object-Detection
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
python object_detector.py
```

4. Abra seu navegador em `http://localhost:5000`

### Uso

#### Interface Web
1. **Upload de Imagem**: Selecione arquivo de imagem para detecção de objetos
2. **Ajustar Configurações**: Defina limite de confiança e parâmetros do modelo
3. **Processar**: Clique em detectar para analisar a imagem
4. **Ver Resultados**: Veja objetos detectados com caixas delimitadoras e rótulos
5. **Download**: Salve imagem anotada com resultados da detecção

#### Endpoints da API

**Detecção de Imagem**
```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@sua_imagem.jpg" \
  -F "confidence=0.5"
```

**Processamento em Lote**
```bash
curl -X POST http://localhost:5000/api/batch_detect \
  -F "images=@imagem1.jpg" \
  -F "images=@imagem2.jpg"
```

#### API Python
```python
from object_detector import ObjectDetector

detector = ObjectDetector()

# Carregar e processar imagem
image_path = "imagem_exemplo.jpg"
detections = detector.detect_objects(image_path)

# Exibir resultados
for detection in detections:
    print(f"Objeto: {detection['class']}")
    print(f"Confiança: {detection['confidence']:.2f}")
    print(f"Caixa Delimitadora: {detection['bbox']}")
```

### Objetos Suportados
O sistema pode detectar mais de 80 classes de objetos incluindo:
- **Pessoas**: Detecção e rastreamento de pessoas
- **Veículos**: Carros, caminhões, motocicletas, bicicletas
- **Animais**: Cães, gatos, pássaros, cavalos
- **Objetos**: Cadeiras, mesas, laptops, telefones
- **Comida**: Frutas, vegetais, bebidas
- **Esportes**: Bolas, equipamentos, atividades

### Modelos de Detecção

#### YOLO (You Only Look Once)
- Detecção rápida em tempo real
- Avaliação de rede neural única
- Bom equilíbrio entre velocidade e precisão

#### SSD (Single Shot Detector)
- Detecção de características multi-escala
- Eficiente para deployment móvel
- Otimizado para velocidade

#### Modelos Personalizados
- Treine com seus próprios datasets
- Ajuste fino para casos de uso específicos
- Detecção de objetos específicos do domínio

### Otimização de Performance
- **Aceleração GPU**: Suporte CUDA para processamento mais rápido
- **Quantização de Modelo**: Tamanho reduzido para deployment móvel
- **Processamento em Lote**: Análise eficiente de múltiplas imagens
- **Cache**: Cache inteligente para detecções repetidas

### Configuração
Personalize parâmetros de detecção em `config.json`:
```json
{
  "confidence_threshold": 0.5,
  "nms_threshold": 0.4,
  "input_size": 416,
  "model_path": "models/yolo.weights"
}
```

### Contribuindo
1. Faça um fork do repositório
2. Crie uma branch de feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adicionar nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

### Licença
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

