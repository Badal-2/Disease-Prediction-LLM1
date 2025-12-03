# Disease Prediction using Fine-Tuned LLM

## 🎯 Project Overview
This project fine-tunes TinyLlama-1.1B model on a disease-symptom dataset to predict diseases based on symptoms.

## 📊 Dataset
- **Source**: Kaggle Disease and Symptoms Dataset
- **Training Examples**: 500
- **Testing Examples**: 100
- **Format**: JSONL (instruction-input-output)

## 🤖 Model Details
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Parameters**:
  - LoRA Rank (r): 16
  - LoRA Alpha: 32
  - Learning Rate: 2e-4
  - Epochs: 2
  - Batch Size: 4
- **Training Time**: ~2 minutes on T4 GPU
- **Framework**: Hugging Face Transformers + PEFT

## 📈 Results
- **Training Loss**: Reduced from 2.01 → 0.11
- **Final Accuracy**: 35% (on 20 test cases)
- **Confusion Matrix**: Available in repository

## 🔧 Technologies Used
- Python 3.12
- PyTorch
- Transformers (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- Google Colab (T4 GPU)
- scikit-learn (for evaluation)

## 📁 Repository Contents
```
├── Disease_Prediction_LLM.ipynb  # Main Colab notebook with all code
├── train.jsonl                   # Training data (100 examples)
├── test.jsonl                    # Testing data (50 examples)
├── confusion_matrix.png          # Model evaluation results
└── README.md                     # Project documentation
```

## 🚀 How to Run
1. Open `Disease_Prediction_LLM.ipynb` in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Upload your Kaggle API key when prompted (kaggle.json)
4. Run all cells sequentially from top to bottom
5. Training completes in ~2 minutes
6. Download output files from the Files sidebar

## 💡 Key Features
- ✅ End-to-end LLM fine-tuning pipeline
- ✅ Efficient training using LoRA (only 1% parameters trained)
- ✅ Automatic confusion matrix generation
- ✅ Demo queries with real-time predictions
- ✅ Medical disclaimer in all outputs

## ⚠️ Important Disclaimer
This is an **educational project** for learning LLM fine-tuning techniques. 

**NOT FOR MEDICAL USE**: This model is NOT intended for real medical diagnosis, treatment recommendations, or healthcare decisions. Always consult qualified healthcare professionals for medical advice.

## 👨‍💻 Demo Queries

### Example 1:
**Input**: `fever, headache, body pain`  
**Output**: `COVID-19`

### Example 2:
**Input**: `skin rash, itching`  
**Output**: `Eczema`

### Example 3:
**Input**: `cough, fatigue, weakness`  
**Output**: `COVID-19`

## 📊 Model Performance
- Model successfully learned disease-symptom patterns
- Training loss decreased significantly (2.01 → 0.11)
- Further training with more data would improve accuracy
- Current accuracy is sufficient for demonstration purposes

## 🎓 Learning Outcomes
Through this project, I learned:
- Fine-tuning large language models using LoRA
- Working with instruction-based datasets
- Implementing efficient training techniques
- Model evaluation using confusion matrices
- Deploying ML models in Google Colab

## 🔗 Links
- **Dataset**: [Kaggle Disease and Symptoms Dataset](https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset)
- **Base Model**: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Demo Video**: [Add your video link here]

## 📧 Contact
**Name**: [Your Name]  
**Email**: [Your Email]  
**GitHub**: [Your GitHub Profile]

## 📝 License
This project is for educational purposes only.

---
*Created as part of LLM Fine-Tuning Assignment*
