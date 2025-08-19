import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import torch

def download_models():
    print("Modeller indiriliyor...")
    
    # MT5 modelini indir
    try:
        print("MT5 modeli indiriliyor...")
        model_name = "Mert1315/Tr-grammer-mt5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Modelleri kaydet
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mt5")
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"MT5 modeli kaydedildi: {save_path}")
    except Exception as e:
        print(f"MT5 model indirme hatası: {str(e)}")
    
    # BERT modelini indir
    try:
        print("BERT modeli indiriliyor...")
        model_name = "dbmdz/bert-base-turkish-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Modelleri kaydet
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bert")
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"BERT modeli kaydedildi: {save_path}")
    except Exception as e:
        print(f"BERT model indirme hatası: {str(e)}")

if __name__ == "__main__":
    download_models() 