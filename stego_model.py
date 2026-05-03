import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stego_model import StegoModel
import os

class PhishingFusion:
    def __init__(self):
        # ---------- 1. URL Model (scikit-learn) ----------
        self.url_model = joblib.load('models/url_model.pkl')
        
        # ---------- 2. XLM-R Text Model (multilingual) ----------
        model_path = 'models/best_xlrm_multilingual_model'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.text_model.eval()
        
        # ---------- 3. Steganography Model ----------
        self.stego_model = StegoModel('models/trpsteg_best.pth')
        
        # Weights for final score (tune based on validation)
        self.text_weight = 0.4
        self.url_weight = 0.4
        self.stego_weight = 0.2
        self.threshold = 0.5   # classification threshold

    def predict_text(self, text):
        """Return phishing probability from email body"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)[0][1].item()  # assuming label 1 = phishing
        return prob

    def predict_urls(self, urls):
        """
        If multiple URLs, take the maximum risk.
        Your url_model.pkl should be a classifier with .predict_proba().
        """
        if not urls:
            return 0.0
        max_prob = 0.0
        for url in urls:
            # Assume model expects a single feature vector – adjust feature extraction!
            # Here we assume the model is a simple CountVectorizer + classifier.
            # Replace with your actual preprocessing pipeline.
            prob = self.url_model.predict_proba([url])[0][1]  # phishing class prob
            max_prob = max(max_prob, prob)
        return max_prob

    def predict_stego(self, attachment_paths):
        """Check images for steganography"""
        if not attachment_paths:
            return 0.0
        max_prob = 0.0
        for path in attachment_paths:
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                prob = self.stego_model.predict(path)
                max_prob = max(max_prob, prob)
        return max_prob

    def fuse_predictions(self, text, urls, attachment_paths):
        """Weighted average of all three scores"""
        text_score = self.predict_text(text)
        url_score = self.predict_urls(urls)
        stego_score = self.predict_stego(attachment_paths)
        
        final_score = (self.text_weight * text_score +
                       self.url_weight * url_score +
                       self.stego_weight * stego_score)
        is_phishing = final_score >= self.threshold
        
        return {
            'phishing_probability': final_score,
            'is_phishing': is_phishing,
            'scores': {
                'text': text_score,
                'url': url_score,
                'stego': stego_score
            }
        }