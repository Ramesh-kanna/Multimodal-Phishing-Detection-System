"""
Phishing Detection System - Main Application
Multi-modal detection using text, URL, and image analysis
"""

import os
import re
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import urllib.parse
import warnings
import cv2
import pytesseract
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')

# =========================================================
# CONFIGURATION
# =========================================================

TEXT_MODEL_PATH = "models/best_xlmr_multilingual_model"
STEGO_MODEL_PATH = "models/trpsteg_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# INITIALIZE FLASK
# =========================================================
app = Flask(__name__)

# =========================================================
# GLOBAL VARIABLES
# =========================================================
text_model = None
tokenizer = None
stego_model = None

# =========================================================
# URL ANALYZER - NO PICKLE DEPENDENCY
# =========================================================

class URLPhishingDetector:
    """Standalone URL phishing detector - no external model files needed"""
    
    def __init__(self):
        self.suspicious_tlds = [
            '.xyz', '.top', '.club', '.online', '.site', '.work', 
            '.date', '.men', '.loan', '.download', '.win', '.bid',
            '.trade', '.webcam', '.review', '.science', '.accountant'
        ]
        
        self.suspicious_keywords = [
            'login', 'secure', 'account', 'verify', 'update', 'bank',
            'paypal', 'apple', 'microsoft', 'amazon', 'signin', 'auth',
            'password', 'credential', 'confirm', 'ssn', 'socialsecurity',
            'ebay', 'netflix', 'chase', 'wellsfargo', 'bankofamerica'
        ]
        
        self.trusted_domains = [
            'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
            'apple.com', 'paypal.com', 'twitter.com', 'linkedin.com',
            'github.com', 'youtube.com', 'netflix.com', 'spotify.com'
        ]
    
    def analyze(self, url):
        """Analyze URL and return comprehensive risk assessment"""
        try:
            if not url or not url.strip():
                return {
                    'score': 0.0,
                    'probability': 0.0,
                    'risk_level': 'LOW',
                    'reasons': ['No URL provided'],
                    'features': {}
                }
            
            # Normalize URL
            original_url = url
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            risk_score = 0.0
            reasons = []
            features = {}
            
            # === FEATURE 1: URL Length ===
            url_length = len(original_url)
            features['url_length'] = url_length
            if url_length > 100:
                risk_score += 0.25
                reasons.append("URL length exceeds 100 characters")
            elif url_length > 75:
                risk_score += 0.15
                reasons.append("URL length exceeds 75 characters")
            
            # === FEATURE 2: IP Address ===
            ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            has_ip = 1 if re.search(ip_pattern, url) else 0
            features['has_ip'] = has_ip
            if has_ip:
                risk_score += 0.35
                reasons.append("IP address used instead of domain name")
            
            # === FEATURE 3: Suspicious TLD ===
            suspicious_tld = False
            for tld in self.suspicious_tlds:
                if tld in domain:
                    suspicious_tld = True
                    risk_score += 0.20
                    reasons.append(f"Suspicious TLD: {tld}")
                    break
            features['suspicious_tld'] = suspicious_tld
            
            # === FEATURE 4: @ Symbol ===
            has_at = 1 if '@' in original_url else 0
            features['has_at'] = has_at
            if has_at:
                risk_score += 0.30
                reasons.append("URL contains '@' symbol - possible credential theft")
            
            # === FEATURE 5: HTTPS ===
            is_https = 1 if parsed.scheme == 'https' else 0
            features['is_https'] = is_https
            if not is_https:
                risk_score += 0.15
                reasons.append("No HTTPS encryption")
            
            # === FEATURE 6: Multiple Dots/Subdomains ===
            dot_count = domain.count('.')
            features['dot_count'] = dot_count
            if dot_count > 4:
                risk_score += 0.20
                reasons.append("Excessive number of subdomains")
            elif dot_count > 3:
                risk_score += 0.10
            
            # === FEATURE 7: Hyphens ===
            hyphen_count = domain.count('-')
            features['hyphen_count'] = hyphen_count
            if hyphen_count > 2:
                risk_score += 0.15
                reasons.append("Multiple hyphens in domain name")
            
            # === FEATURE 8: Suspicious Keywords ===
            keyword_matches = []
            for keyword in self.suspicious_keywords:
                if keyword in url.lower():
                    keyword_matches.append(keyword)
                    risk_score += 0.12
            
            if keyword_matches:
                features['suspicious_keywords'] = keyword_matches[:3]
                reasons.append(f"Suspicious keywords: {', '.join(keyword_matches[:2])}")
            
            # === FEATURE 9: Trusted Domain Check ===
            is_trusted = False
            for trusted in self.trusted_domains:
                if trusted in domain:
                    is_trusted = True
                    risk_score *= 0.3  # Reduce risk significantly
                    reasons.append(f"Trusted domain detected: {trusted}")
                    break
            features['is_trusted'] = is_trusted
            
            # === FEATURE 10: Numeric Characters ===
            digit_count = sum(c.isdigit() for c in url)
            features['digit_count'] = digit_count
            if digit_count > 10 and not has_ip:
                risk_score += 0.15
                reasons.append("Unusual number of digits in URL")
            
            # === FEATURE 11: URL Shorteners ===
            shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'short.link']
            for shortener in shorteners:
                if shortener in domain:
                    risk_score += 0.10
                    reasons.append(f"URL shortener detected: {shortener}")
                    break
            
            # === FEATURE 12: Double Slash Redirect ===
            if '//' in parsed.path:
                risk_score += 0.20
                reasons.append("Double slash redirect detected")
            
            # Normalize score to 0-1
            final_score = min(1.0, risk_score)
            
            # Determine risk level
            if final_score >= 0.65:
                risk_level = 'HIGH'
            elif final_score >= 0.40:
                risk_level = 'MEDIUM'
            elif final_score >= 0.20:
                risk_level = 'LOW'
            else:
                risk_level = 'NEGLIGIBLE'
            
            return {
                'score': round(final_score, 3),
                'probability': final_score,
                'risk_level': risk_level,
                'reasons': reasons[:5],  # Top 5 reasons
                'features': features,
                'url': original_url[:100]
            }
            
        except Exception as e:
            print(f"URL analysis error: {e}")
            return {
                'score': 0.5,
                'probability': 0.5,
                'risk_level': 'MEDIUM',
                'reasons': ['Analysis error - using default risk'],
                'features': {},
                'url': url[:100]
            }

# =========================================================
# TEXT ANALYZER - NO TRANSFORMER DEPENDENCY
# =========================================================

class TextPhishingDetector:
    """Standalone text phishing detector - heuristic-based"""
    
    def __init__(self):
        self.phishing_patterns = [
            (r'verify.*account', 0.25, 'Account verification request'),
            (r'update.*payment', 0.25, 'Payment update request'),
            (r'confirm.*identity', 0.30, 'Identity confirmation required'),
            (r'security.*alert', 0.25, 'Security alert'),
            (r'suspend.*account', 0.35, 'Account suspension threat'),
            (r'limited.*account', 0.30, 'Account limitation notice'),
            (r'unusual.*activity', 0.25, 'Unusual activity detected'),
            (r'click.*link.*verify', 0.35, 'Request to click verification link'),
            (r'password.*expir', 0.25, 'Password expiration notice'),
            (r'bank.*detail', 0.30, 'Request for bank details'),
            (r'credit.*card.*verify', 0.35, 'Credit card verification'),
            (r'social.*security', 0.40, 'SSN/social security number request'),
            (r'urgent.*action', 0.30, 'Urgent action required'),
            (r'account.*blocked', 0.35, 'Account blocked notification'),
            (r'immediately', 0.20, 'Urgent/immediate action required'),
            (r'verify.*now', 0.25, 'Immediate verification required'),
            (r'click.*below', 0.20, 'Click link below'),
            (r'copy.*link', 0.20, 'Copy and paste link')
        ]
        
        self.urgent_words = [
            'urgent', 'immediately', 'warning', 'alert', 'suspended',
            'limited', 'blocked', 'restricted', 'deactivated', 'expires',
            'expiring', 'terminated', 'compromised', 'unauthorized'
        ]
        
        self.request_words = [
            'verify', 'confirm', 'update', 'validate', 'secure',
            'restore', 'reactivate', 'unlock', 'reset', 'recover'
        ]
        
        self.personal_info_words = [
            'ssn', 'social security', 'credit card', 'debit card',
            'bank account', 'routing number', 'atm pin', 'password',
            'username', 'login', 'credentials', 'date of birth'
        ]
    
    def analyze(self, text):
        """Analyze text and return phishing risk assessment"""
        try:
            if not text or len(text.strip()) < 15:
                return {
                    'score': 0.0,
                    'probability': 0.0,
                    'risk_level': 'LOW',
                    'reasons': ['Insufficient text for analysis'],
                    'matches': []
                }
            
            text_lower = text.lower()
            risk_score = 0.0
            reasons = []
            matches = []
            
            # Check phishing patterns
            for pattern, weight, description in self.phishing_patterns:
                if re.search(pattern, text_lower):
                    risk_score += weight
                    matches.append(pattern)
                    if len(reasons) < 4:
                        reasons.append(description)
            
            # Check urgent words
            urgent_count = 0
            for word in self.urgent_words:
                if word in text_lower:
                    urgent_count += 1
                    if urgent_count <= 3:
                        risk_score += 0.10
            
            if urgent_count > 0:
                reasons.append(f"Urgent language detected ({urgent_count} instances)")
            
            # Check request words
            request_count = 0
            for word in self.request_words:
                if word in text_lower:
                    request_count += 1
                    if request_count <= 3:
                        risk_score += 0.08
            
            if request_count > 0:
                reasons.append(f"Action request detected ({request_count} instances)")
            
            # Check personal info requests
            for word in self.personal_info_words:
                if word in text_lower:
                    risk_score += 0.20
                    reasons.append(f"Request for {word}")
                    break
            
            # Check for urgency indicators
            urgency_indicators = ['!', '!!!', 'ASAP', 'immediate', 'right now']
            for indicator in urgency_indicators:
                if indicator.lower() in text_lower:
                    risk_score += 0.05
                    break
            
            # Check for grammatical errors (simplified)
            if re.search(r'\b[Ii]mportant\b', text_lower):
                risk_score += 0.10
                reasons.append("Spelling errors detected")
            
            # Check for generic greetings
            if 'dear customer' in text_lower or 'dear user' in text_lower:
                risk_score += 0.15
                reasons.append("Generic greeting (not personalized)")
            
            # Normalize score
            final_score = min(1.0, risk_score)
            
            # Text length factor - very short texts get reduced score
            if len(text) < 50:
                final_score *= 0.7
            
            # Determine risk level
            if final_score >= 0.60:
                risk_level = 'HIGH'
            elif final_score >= 0.35:
                risk_level = 'MEDIUM'
            elif final_score >= 0.15:
                risk_level = 'LOW'
            else:
                risk_level = 'NEGLIGIBLE'
            
            return {
                'score': round(final_score, 3),
                'probability': final_score,
                'risk_level': risk_level,
                'reasons': reasons[:5],
                'matches': matches[:5],
                'text_length': len(text)
            }
            
        except Exception as e:
            print(f"Text analysis error: {e}")
            return {
                'score': 0.5,
                'probability': 0.5,
                'risk_level': 'MEDIUM',
                'reasons': ['Analysis error - using default risk'],
                'matches': []
            }

# =========================================================
# IMAGE ANALYZER
# =========================================================

class ImagePhishingDetector:
    """Image analysis for phishing detection"""
    
    def __init__(self):
        self.brand_logos = {
            'paypal': ['paypal', 'pay pal'],
            'apple': ['apple', 'icloud', 'itunes'],
            'microsoft': ['microsoft', 'outlook', 'office365'],
            'amazon': ['amazon', 'amzn'],
            'google': ['google', 'gmail', 'youtube'],
            'facebook': ['facebook', 'meta'],
            'netflix': ['netflix'],
            'chase': ['chase', 'jp morgan'],
            'wellsfargo': ['wells fargo', 'wellsfargo'],
            'bankofamerica': ['bank of america', 'bofa']
        }
    
    def extract_text(self, image):
        """Extract text from image using OCR"""
        try:
            # Convert PIL to OpenCV
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocessing for better OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            denoised = cv2.medianBlur(thresh, 3)
            
            # OCR
            text = pytesseract.image_to_string(denoised)
            return text.strip()
        except:
            # Fallback to direct OCR
            try:
                return pytesseract.image_to_string(image).strip()
            except:
                return ""
    
    def detect_logos(self, image):
        """Detect brand logos in image (simplified)"""
        # This is a placeholder - in production, use a proper logo detection model
        return []

# =========================================================
# FUSION ENGINE
# =========================================================

class FusionEngine:
    """Multi-modal fusion engine for phishing detection"""
    
    def __init__(self):
        self.weights = {
            'text': 0.45,
            'url': 0.35,
            'image': 0.20
        }
        
        self.thresholds = {
            'high': 0.70,
            'medium': 0.50,
            'low': 0.30
        }
    
    def fuse(self, text_result=None, url_result=None, stego_prob=None):
        """
        Fuse multiple detection results into final verdict
        
        Args:
            text_result: Dictionary from TextPhishingDetector.analyze()
            url_result: Dictionary from URLPhishingDetector.analyze()
            stego_prob: Float probability from stego detector
        """
        
        scores = []
        weights = []
        evidence = []
        
        # === Process Text Result ===
        if text_result and 'probability' in text_result:
            text_score = text_result['probability']
            scores.append(text_score)
            weights.append(self.weights['text'])
            
            if text_score > 0.6:
                evidence.append(f"Text: {text_result.get('risk_level', 'HIGH')} risk")
                if text_result.get('reasons'):
                    evidence.extend(text_result['reasons'][:2])
        
        # === Process URL Result ===
        if url_result and 'probability' in url_result:
            url_score = url_result['probability']
            scores.append(url_score)
            weights.append(self.weights['url'])
            
            if url_score > 0.6:
                evidence.append(f"URL: {url_result.get('risk_level', 'HIGH')} risk")
                if url_result.get('reasons'):
                    evidence.extend(url_result['reasons'][:2])
        
        # === Process Stego Result ===
        if stego_prob is not None:
            scores.append(stego_prob)
            weights.append(self.weights['image'])
            
            if stego_prob > 0.65:
                evidence.append("Image: Hidden data detected")
            elif stego_prob > 0.45:
                evidence.append("Image: Suspicious patterns detected")
        
        # === Calculate Final Score ===
        if not scores:
            return {
                'score': 0.0,
                'verdict': 'NO_DATA',
                'risk_level': 'UNKNOWN',
                'confidence': 0.0,
                'evidence': ['No input data provided'],
                'recommendation': 'Provide text, URL, or image for analysis'
            }
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Apply confidence boost/penalty
        confidence = self._calculate_confidence(scores, weights)
        
        # Adjust score based on conflicting signals
        final_score = self._adjust_for_conflicts(final_score, scores)
        
        # Determine verdict and risk level
        verdict, risk_level = self._get_verdict(final_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(final_score, risk_level, evidence)
        
        # Remove duplicates from evidence
        evidence = list(dict.fromkeys(evidence))[:6]  # Top 6 unique evidence points
        
        return {
            'score': round(final_score, 4),
            'verdict': verdict,
            'risk_level': risk_level,
            'confidence': round(confidence, 4),
            'evidence': evidence,
            'recommendation': recommendation,
            'scores': {
                'text': round(text_result.get('probability', 0), 4) if text_result else None,
                'url': round(url_result.get('probability', 0), 4) if url_result else None,
                'stego': round(stego_prob, 4) if stego_prob is not None else None
            }
        }
    
    def _calculate_confidence(self, scores, weights):
        """Calculate confidence in the prediction"""
        if not scores:
            return 0.0
        
        # More signals = higher confidence
        signal_confidence = min(1.0, len(scores) / 3)
        
        # Extreme scores = higher confidence
        avg_score = sum(scores) / len(scores)
        extremity = abs(avg_score - 0.5) * 2
        
        # Consistency = higher confidence
        if len(scores) > 1:
            consistency = 1.0 - (max(scores) - min(scores))
        else:
            consistency = 0.8
        
        confidence = (signal_confidence * 0.3 + extremity * 0.4 + consistency * 0.3)
        return min(1.0, confidence)
    
    def _adjust_for_conflicts(self, final_score, scores):
        """Adjust score when there are conflicting signals"""
        if len(scores) > 1:
            # Check for conflicting signals (both high and low)
            high_count = sum(1 for s in scores if s > 0.6)
            low_count = sum(1 for s in scores if s < 0.4)
            
            if high_count > 0 and low_count > 0:
                # Conflicting signals - reduce confidence but keep score
                final_score = final_score * 0.9 + 0.1  # Slight bias toward caution
        
        return min(1.0, max(0.0, final_score))
    
    def _get_verdict(self, score):
        """Determine verdict and risk level based on score"""
        if score >= self.thresholds['high']:
            return 'PHISHING', 'HIGH'
        elif score >= self.thresholds['medium']:
            return 'SUSPICIOUS', 'MEDIUM'
        elif score >= self.thresholds['low']:
            return 'LOW_RISK', 'LOW'
        else:
            return 'LEGITIMATE', 'NEGLIGIBLE'
    
    def _generate_recommendation(self, score, risk_level, evidence):
        """Generate user recommendation based on analysis"""
        if score >= 0.7:
            return "🚨 DO NOT interact with this content. This is likely a phishing attempt."
        elif score >= 0.5:
            return "⚠️ Exercise caution. Verify the source through official channels before proceeding."
        elif score >= 0.3:
            return "🔍 Some suspicious elements detected. Review carefully before trusting."
        else:
            return "✅ No significant threats detected. Normal interaction should be safe."

# =========================================================
# INITIALIZE COMPONENTS
# =========================================================

print("\n" + "="*60)
print("🛡️  PHISHING DETECTION SYSTEM - MULTI-MODAL")
print("="*60)

# Initialize detectors
print("\n📦 Initializing detection modules...")
url_detector = URLPhishingDetector()
print("  ✅ URL Phishing Detector initialized")

text_detector = TextPhishingDetector()
print("  ✅ Text Phishing Detector initialized")

fusion_engine = FusionEngine()
print("  ✅ Fusion Engine initialized")

image_detector = ImagePhishingDetector()
print("  ✅ Image Analyzer initialized")

# Try to load optional models
print("\n🔧 Loading optional models...")

# Try to load transformer model
try:
    print("  • Attempting to load transformer model...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
    text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
    text_model.to(DEVICE)
    text_model.eval()
    print("  ✅ Transformer model loaded successfully")
except Exception as e:
    print(f"  ⚠ Transformer model not loaded: {e}")
    print("  • Using heuristic text detector")

# Try to load stego model
try:
    print("  • Attempting to load stego detection model...")
    from stego_model import load_stego_model, predict_stego
    stego_model = load_stego_model(STEGO_MODEL_PATH)
    print("  ✅ Stego model loaded successfully")
except Exception as e:
    print(f"  ⚠ Stego model not loaded: {e}")
    stego_model = None

print("\n" + "="*60)
print("✅ SYSTEM READY - Listening for requests")
print("="*60 + "\n")

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================

def predict_text_heuristic(text):
    """Use heuristic text detector"""
    return text_detector.analyze(text)

def predict_text_transformer(text):
    """Use transformer model if available"""
    try:
        if text_model is None or tokenizer is None:
            return predict_text_heuristic(text)
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            phishing_prob = probs[0][1].item()
        
        return {
            'score': phishing_prob,
            'probability': phishing_prob,
            'risk_level': 'HIGH' if phishing_prob > 0.6 else 'MEDIUM' if phishing_prob > 0.35 else 'LOW',
            'reasons': ['Transformer model prediction'],
            'matches': []
        }
    except Exception as e:
        print(f"Transformer prediction error: {e}")
        return predict_text_heuristic(text)

def predict_url(url):
    """Predict phishing probability from URL"""
    return url_detector.analyze(url)

def predict_image(file):
    """Process image and extract information"""
    try:
        img = Image.open(file.stream).convert("RGB")
        
        # Extract text via OCR
        ocr_text = image_detector.extract_text(img)
        
        # Stego detection
        stego_prob = None
        if stego_model is not None:
            try:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                tensor = transform(img).unsqueeze(0)
                stego_result = predict_stego(stego_model, tensor)
                stego_prob = stego_result.get("probability", 0.5)
            except:
                stego_prob = 0.3
        
        # Analyze OCR text if present
        text_result = None
        if ocr_text and len(ocr_text.strip()) > 20:
            if text_model is not None:
                text_result = predict_text_transformer(ocr_text)
            else:
                text_result = predict_text_heuristic(ocr_text)
        
        return {
            'ocr_text': ocr_text[:500],  # Limit length
            'stego_prob': stego_prob,
            'text_result': text_result
        }
    except Exception as e:
        print(f"Image processing error: {e}")
        return {
            'ocr_text': '',
            'stego_prob': 0.3,
            'text_result': None,
            'error': str(e)
        }

# =========================================================
# FLASK ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        text_result = None
        url_result = None
        stego_prob = None
        ocr_text = None
        
        # ---------- Process Text Input ----------
        text = request.form.get("text")
        if text and text.strip():
            if text_model is not None:
                text_result = predict_text_transformer(text)
            else:
                text_result = predict_text_heuristic(text)
            print(f"  📝 Text analysis: {text_result['risk_level']} risk ({text_result['score']:.3f})")
        
        # ---------- Process URL Input ----------
        url = request.form.get("url")
        if url and url.strip():
            url_result = predict_url(url)
            print(f"  🔗 URL analysis: {url_result['risk_level']} risk ({url_result['score']:.3f})")
        
        # ---------- Process Image Input ----------
        file = request.files.get("image")
        if file and file.filename != "":
            image_result = predict_image(file)
            stego_prob = image_result.get('stego_prob')
            ocr_text = image_result.get('ocr_text')
            
            if ocr_text:
                print(f"  🖼️  OCR extracted: {len(ocr_text)} chars")
            
            # Use OCR text for analysis if no text was provided
            if ocr_text and not text:
                if text_model is not None:
                    text_result = predict_text_transformer(ocr_text)
                else:
                    text_result = predict_text_heuristic(ocr_text)
                print(f"  📝 OCR text analysis: {text_result['risk_level']} risk")
        
        # ---------- Fusion ----------
        fusion_result = fusion_engine.fuse(
            text_result=text_result,
            url_result=url_result,
            stego_prob=stego_prob
        )
        
        print(f"  🎯 Fusion verdict: {fusion_result['verdict']} ({fusion_result['score']:.3f})")
        
        # ---------- Prepare Response ----------
        response = {
            'verdict': fusion_result['verdict'],
            'risk_level': fusion_result['risk_level'],
            'confidence': fusion_result['confidence'],
            'score': fusion_result['score'],
            'evidence': fusion_result['evidence'],
            'recommendation': fusion_result['recommendation'],
            'scores': fusion_result['scores']
        }
        
        if ocr_text:
            response['ocr_text'] = ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            'verdict': 'ERROR',
            'risk_level': 'UNKNOWN',
            'confidence': 0.0,
            'score': 0.0,
            'evidence': [f'Analysis failed: {str(e)}'],
            'recommendation': 'Please try again with valid input',
            'error': str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'models': {
            'text_transformer': text_model is not None,
            'text_heuristic': True,
            'url_heuristic': True,
            'stego': stego_model is not None,
            'fusion': True
        },
        'version': '2.0.0'
    })

@app.route("/analyze/url", methods=["POST"])
def analyze_url_only():
    """Standalone URL analysis"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        result = url_detector.analyze(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/analyze/text", methods=["POST"])
def analyze_text_only():
    """Standalone text analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if text_model is not None:
            result = predict_text_transformer(text)
        else:
            result = predict_text_heuristic(text)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# =========================================================
# CREATE DEFAULT TEMPLATE
# =========================================================

def create_default_template():
    """Create a basic index.html if it doesn't exist"""
    template_dir = os.path.join(app.root_path, 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(template_path):
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
            font-size: 2.2em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .input-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
            font-size: 1em;
        }
        
        input, textarea {
            width: 100%;
            padding: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s;
            font-family: inherit;
        }
        
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .file-input {
            border: 2px dashed #e0e0e0;
            padding: 25px;
            text-align: center;
            cursor: pointer;
            border-radius: 12px;
            background: #fafafa;
        }
        
        .file-input:hover {
            border-color: #667eea;
            background: #f5f5f5;
        }
        
        .file-input.has-file {
            border-color: #4caf50;
            background: #e8f5e9;
        }
        
        button {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102,126,234,0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 40px;
            padding: 25px;
            border-radius: 16px;
            display: none;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result.show {
            display: block;
        }
        
        .verdict-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        .verdict-title {
            font-size: 1.8em;
            font-weight: 700;
        }
        
        .risk-badge {
            padding: 8px 20px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 1em;
            color: white;
        }
        
        .confidence-meter {
            margin: 20px 0;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 12px;
        }
        
        .meter-bar {
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .meter-fill {
            height: 100%;
            width: 0%;
            transition: width 0.5s;
            border-radius: 5px;
        }
        
        .evidence-list {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 12px;
        }
        
        .evidence-item {
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
        }
        
        .evidence-item:last-child {
            border-bottom: none;
        }
        
        .scores-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .score-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .score-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .score-value {
            font-size: 1.8em;
            font-weight: 700;
            color: #333;
        }
        
        .recommendation {
            margin-top: 20px;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 12px;
            font-size: 1.1em;
            color: #0c5460;
            border-left: 5px solid #2196f3;
        }
        
        .ocr-text {
            margin-top: 20px;
            padding: 15px;
            background: #fff3e0;
            border-radius: 12px;
            font-size: 0.9em;
            color: #856404;
            border-left: 5px solid #ff9800;
        }
        
        .phishing { background: #ffebee; }
        .suspicious { background: #fff3e0; }
        .low-risk { background: #e8f5e9; }
        .legitimate { background: #e8eaf6; }
        .error { background: #ffebee; }
        
        .phishing .meter-fill { background: #f44336; }
        .suspicious .meter-fill { background: #ff9800; }
        .low-risk .meter-fill { background: #4caf50; }
        .legitimate .meter-fill { background: #2196f3; }
        
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Phishing Detection System</h1>
        <div class="subtitle">Multi-modal analysis using text, URL, and image forensics</div>
        
        <div class="input-group">
            <label for="text">📝 Text Content</label>
            <textarea id="text" placeholder="Paste suspicious email, message, or webpage text here..."></textarea>
        </div>
        
        <div class="input-group">
            <label for="url">🔗 URL</label>
            <input type="url" id="url" placeholder="Enter suspicious URL (e.g., https://example.com)">
        </div>
        
        <div class="input-group">
            <label for="image">🖼️ Image</label>
            <div class="file-input" id="file-input" onclick="document.getElementById('image').click()">
                <span id="file-label">📁 Click to select or drag image</span>
            </div>
            <input type="file" id="image" accept="image/*" style="display: none;">
        </div>
        
        <button onclick="analyze()">🔍 Analyze</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing with multi-modal AI...</p>
        </div>
        
        <div id="result" class="result">
            <div class="verdict-header">
                <span class="verdict-title" id="verdict"></span>
                <span class="risk-badge" id="risk-badge"></span>
            </div>
            
            <div class="confidence-meter" id="confidence-meter">
                <div style="display: flex; justify-content: space-between;">
                    <span>Confidence</span>
                    <span id="confidence-value">0%</span>
                </div>
                <div class="meter-bar">
                    <div class="meter-fill" id="meter-fill" style="width: 0%;"></div>
                </div>
            </div>
            
            <div class="scores-grid" id="scores-grid"></div>
            
            <div class="evidence-list" id="evidence-list">
                <h4 style="margin-bottom: 10px; color: #333;">🔍 Detection Evidence</h4>
                <div id="evidence-items"></div>
            </div>
            
            <div class="recommendation" id="recommendation"></div>
            
            <div class="ocr-text" id="ocr-text" style="display: none;"></div>
        </div>
        
        <div class="footer">
            ⚡ Multi-modal detection | Text + URL + Image Forensics
        </div>
    </div>

    <script>
        // File input handler
        document.getElementById('image').addEventListener('change', function(e) {
            const fileInput = document.getElementById('file-input');
            const fileLabel = document.getElementById('file-label');
            
            if (this.files.length > 0) {
                fileInput.classList.add('has-file');
                fileLabel.innerHTML = `📁 ${this.files[0].name}`;
            } else {
                fileInput.classList.remove('has-file');
                fileLabel.innerHTML = '📁 Click to select or drag image';
            }
        });

        async function analyze() {
            const formData = new FormData();
            const text = document.getElementById('text').value;
            const url = document.getElementById('url').value;
            const image = document.getElementById('image').files[0];
            
            if (text) formData.append('text', text);
            if (url) formData.append('url', url);
            if (image) formData.append('image', image);
            
            if (!text && !url && !image) {
                alert('Please enter text, URL, or select an image');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').classList.remove('show');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Show result
                const resultDiv = document.getElementById('result');
                resultDiv.classList.add('show');
                
                // Remove existing classes
                resultDiv.classList.remove('phishing', 'suspicious', 'low-risk', 'legitimate', 'error');
                
                // Set verdict and styling
                const verdict = data.verdict;
                const riskLevel = data.risk_level;
                
                document.getElementById('verdict').innerHTML = getVerdictIcon(verdict) + ' ' + verdict;
                
                // Set risk badge
                const riskBadge = document.getElementById('risk-badge');
                riskBadge.innerHTML = riskLevel + ' RISK';
                riskBadge.style.background = getRiskColor(riskLevel);
                
                // Set confidence
                const confidence = (data.confidence * 100).toFixed(1);
                document.getElementById('confidence-value').innerHTML = confidence + '%';
                document.getElementById('meter-fill').style.width = confidence + '%';
                
                // Set result class
                if (verdict === 'PHISHING') {
                    resultDiv.classList.add('phishing');
                } else if (verdict === 'SUSPICIOUS') {
                    resultDiv.classList.add('suspicious');
                } else if (verdict === 'LOW_RISK') {
                    resultDiv.classList.add('low-risk');
                } else if (verdict === 'LEGITIMATE') {
                    resultDiv.classList.add('legitimate');
                } else {
                    resultDiv.classList.add('error');
                }
                
                // Display scores
                let scoresHtml = '';
                if (data.scores) {
                    if (data.scores.text !== null) {
                        scoresHtml += `<div class="score-card">
                            <div class="score-label">📝 Text</div>
                            <div class="score-value">${(data.scores.text * 100).toFixed(0)}%</div>
                        </div>`;
                    }
                    if (data.scores.url !== null) {
                        scoresHtml += `<div class="score-card">
                            <div class="score-label">🔗 URL</div>
                            <div class="score-value">${(data.scores.url * 100).toFixed(0)}%</div>
                        </div>`;
                    }
                    if (data.scores.stego !== null) {
                        scoresHtml += `<div class="score-card">
                            <div class="score-label">🖼️ Image</div>
                            <div class="score-value">${(data.scores.stego * 100).toFixed(0)}%</div>
                        </div>`;
                    }
                }
                document.getElementById('scores-grid').innerHTML = scoresHtml;
                
                // Display evidence
                let evidenceHtml = '';
                if (data.evidence && data.evidence.length > 0) {
                    data.evidence.forEach(item => {
                        evidenceHtml += `<div class="evidence-item">• ${item}</div>`;
                    });
                } else {
                    evidenceHtml = '<div class="evidence-item">No specific evidence available</div>';
                }
                document.getElementById('evidence-items').innerHTML = evidenceHtml;
                
                // Display recommendation
                document.getElementById('recommendation').innerHTML = data.recommendation || 'No recommendation available';
                
                // Display OCR text if present
                if (data.ocr_text) {
                    const ocrDiv = document.getElementById('ocr-text');
                    ocrDiv.style.display = 'block';
                    ocrDiv.innerHTML = `<strong>📄 Extracted Text from Image:</strong><br>${escapeHtml(data.ocr_text)}`;
                } else {
                    document.getElementById('ocr-text').style.display = 'none';
                }
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                
                const resultDiv = document.getElementById('result');
                resultDiv.classList.add('show', 'error');
                document.getElementById('verdict').innerHTML = '❌ ERROR';
                document.getElementById('risk-badge').innerHTML = 'FAILED';
                document.getElementById('confidence-value').innerHTML = '0%';
                document.getElementById('recommendation').innerHTML = 'Analysis failed. Please try again.';
            }
        }
        
        function getVerdictIcon(verdict) {
            switch(verdict) {
                case 'PHISHING': return '⚠️';
                case 'SUSPICIOUS': return '⚠️';
                case 'LOW_RISK': return 'ℹ️';
                case 'LEGITIMATE': return '✅';
                default: return '❓';
            }
        }
        
        function getRiskColor(risk) {
            switch(risk) {
                case 'HIGH': return '#f44336';
                case 'MEDIUM': return '#ff9800';
                case 'LOW': return '#4caf50';
                case 'NEGLIGIBLE': return '#2196f3';
                default: return '#9e9e9e';
            }
        }
        
        function escapeHtml(unsafe) {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
        
        // Drag and drop
        const fileInput = document.getElementById('file-input');
        
        fileInput.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileInput.style.borderColor = '#667eea';
            fileInput.style.background = '#f0f0f0';
        });
        
        fileInput.addEventListener('dragleave', (e) => {
            e.preventDefault();
            fileInput.style.borderColor = '#e0e0e0';
            fileInput.style.background = '#fafafa';
        });
        
        fileInput.addEventListener('drop', (e) => {
            e.preventDefault();
            fileInput.style.borderColor = '#e0e0e0';
            fileInput.style.background = '#fafafa';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('image').files = files;
                fileInput.classList.add('has-file');
                document.getElementById('file-label').innerHTML = `📁 ${files[0].name}`;
            }
        });
    </script>
</body>
</html>'''
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print("📄 Created default index.html template")

# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":
    # Create default template
    create_default_template()
    
    # Print startup banner
    print("\n" + "="*60)
    print("🌐 WEB SERVER STARTING")
    print("="*60)
    print("📍 URL: http://localhost:5000")
    print("📍 Health check: http://localhost:5000/health")
    print("="*60 + "\n")
    
    # Run app
    app.run(debug=True, host="0.0.0.0", port=5000)