"""
Multi-Modal Fusion Engine for Phishing Detection
Combines text, URL, and image analysis for robust phishing detection
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple


class PhishingFusionEngine:
    """
    Advanced fusion engine for combining multiple phishing detection signals
    
    Features:
    - Weighted ensemble with dynamic confidence adjustment
    - Conflict resolution between modalities
    - Evidence aggregation and explanation generation
    - Risk level classification
    """
    
    def __init__(self):
        # Base weights for different modalities
        self.weights = {
            'text': 0.45,      # Text content analysis
            'url': 0.35,       # URL structure analysis
            'image': 0.20,     # Image steganography detection
        }
        
        # Decision thresholds
        self.thresholds = {
            'critical': 0.80,   # Critical risk - immediate action
            'high': 0.65,       # High risk - strong indicators
            'medium': 0.45,     # Medium risk - suspicious
            'low': 0.25,        # Low risk - minor indicators
            'negligible': 0.10  # Negligible risk - safe
        }
        
        # Confidence calibration
        self.confidence_params = {
            'min_signals': 2,           # Minimum signals for high confidence
            'extremity_factor': 0.4,    # Weight for extreme scores
            'consistency_factor': 0.3,  # Weight for consistent signals
            'signal_count_factor': 0.3  # Weight for number of signals
        }
    
    def fuse(self,
             text_result: Optional[Dict] = None,
             url_result: Optional[Dict] = None,
             stego_prob: Optional[float] = None,
             custom_weights: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fuse multiple detection results into a single verdict
        
        Args:
            text_result: Dictionary from text analysis containing 'probability' and 'risk_level'
            url_result: Dictionary from URL analysis containing 'probability' and 'risk_level'
            stego_prob: Float probability from steganography detection
            custom_weights: Optional custom weights for modalities
            
        Returns:
            Dictionary containing fused verdict, confidence, evidence, and recommendations
        """
        
        # Use custom weights if provided
        weights = custom_weights if custom_weights else self.weights.copy()
        
        # Collect all available signals
        signals = []
        signal_weights = []
        evidence = []
        modality_scores = {}
        
        # === Process Text Signal ===
        if text_result and isinstance(text_result, dict):
            text_score = text_result.get('probability', text_result.get('score', 0.5))
            text_risk = text_result.get('risk_level', 'UNKNOWN')
            
            signals.append(text_score)
            signal_weights.append(weights['text'])
            modality_scores['text'] = text_score
            
            # Add evidence from text analysis
            if text_score > 0.6:
                evidence.append(f"📝 Text: {text_risk} risk ({text_score:.0%})")
                if text_result.get('reasons'):
                    evidence.extend(text_result['reasons'][:2])
        
        # === Process URL Signal ===
        if url_result and isinstance(url_result, dict):
            url_score = url_result.get('probability', url_result.get('score', 0.5))
            url_risk = url_result.get('risk_level', 'UNKNOWN')
            
            signals.append(url_score)
            signal_weights.append(weights['url'])
            modality_scores['url'] = url_score
            
            # Add evidence from URL analysis
            if url_score > 0.6:
                evidence.append(f"🔗 URL: {url_risk} risk ({url_score:.0%})")
                if url_result.get('reasons'):
                    evidence.extend(url_result['reasons'][:2])
        
        # === Process Stego Signal ===
        if stego_prob is not None:
            signals.append(stego_prob)
            signal_weights.append(weights['image'])
            modality_scores['image'] = stego_prob
            
            # Add evidence from stego analysis
            if stego_prob > 0.65:
                evidence.append("🖼️ Image: Hidden data detected (steganography)")
            elif stego_prob > 0.45:
                evidence.append("🖼️ Image: Suspicious patterns detected")
        
        # === Handle No Signals Case ===
        if not signals:
            return self._create_empty_result()
        
        # === Calculate Weighted Score ===
        total_weight = sum(signal_weights)
        weighted_score = sum(s * w for s, w in zip(signals, signal_weights)) / total_weight
        
        # === Calculate Confidence ===
        confidence = self._calculate_confidence(signals, signal_weights)
        
        # === Resolve Conflicts ===
        weighted_score, conflict_detected = self._resolve_conflicts(
            weighted_score, signals, signal_weights
        )
        
        if conflict_detected:
            evidence.append("⚠️ Conflicting signals detected - using cautious weighting")
        
        # === Apply Threshold Adjustments ===
        weighted_score = self._apply_threshold_adjustments(weighted_score, signals)
        
        # === Determine Verdict and Risk Level ===
        verdict, risk_level = self._classify_risk(weighted_score)
        
        # === Generate Recommendation ===
        recommendation = self._generate_recommendation(
            weighted_score, risk_level, len(signals)
        )
        
        # === Clean and Deduplicate Evidence ===
        evidence = self._clean_evidence(evidence)
        
        # === Calculate Individual Risk Levels ===
        individual_risks = self._get_individual_risk_levels(
            text_result, url_result, stego_prob
        )
        
        return {
            'score': round(weighted_score, 4),
            'verdict': verdict,
            'risk_level': risk_level,
            'confidence': round(confidence, 4),
            'evidence': evidence[:8],  # Limit to top 8 evidence points
            'recommendation': recommendation,
            'signals_processed': len(signals),
            'conflict_detected': conflict_detected,
            'modality_scores': modality_scores,
            'individual_risks': individual_risks,
            'weights_used': {k: v for k, v in weights.items() if k in modality_scores}
        }
    
    def _calculate_confidence(self, signals: List[float], weights: List[float]) -> float:
        """Calculate confidence in the fused prediction"""
        if not signals:
            return 0.0
        
        # Signal count confidence (more signals = higher confidence)
        signal_count_conf = min(1.0, len(signals) / 3)
        
        # Extremity confidence (scores far from 0.5 are more confident)
        avg_score = sum(signals) / len(signals)
        extremity_conf = abs(avg_score - 0.5) * 2
        
        # Consistency confidence (agreement between signals)
        if len(signals) > 1:
            # Calculate weighted standard deviation
            weighted_mean = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
            variance = sum(w * (s - weighted_mean) ** 2 for s, w in zip(signals, weights)) / sum(weights)
            std_dev = np.sqrt(variance)
            consistency_conf = 1.0 - min(1.0, std_dev)
        else:
            consistency_conf = 0.8  # Default for single signal
        
        # Weighted combination
        confidence = (
            self.confidence_params['signal_count_factor'] * signal_count_conf +
            self.confidence_params['extremity_factor'] * extremity_conf +
            self.confidence_params['consistency_factor'] * consistency_conf
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _resolve_conflicts(self, 
                          base_score: float, 
                          signals: List[float], 
                          weights: List[float]) -> Tuple[float, bool]:
        """Resolve conflicts between different modalities"""
        if len(signals) < 2:
            return base_score, False
        
        # Check for strong conflicts (one high, one low)
        high_signals = [s for s in signals if s > 0.65]
        low_signals = [s for s in signals if s < 0.35]
        
        if high_signals and low_signals:
            # Conflict detected - apply cautious weighting
            # Move score slightly toward caution (higher risk)
            adjusted_score = base_score * 0.85 + 0.15
            return min(1.0, adjusted_score), True
        
        return base_score, False
    
    def _apply_threshold_adjustments(self, score: float, signals: List[float]) -> float:
        """Apply non-linear adjustments near thresholds"""
        # Boost scores near high threshold
        if 0.60 <= score <= 0.70:
            score = score * 1.05  # 5% boost in uncertain region
        
        # Reduce scores near low threshold
        if 0.30 <= score <= 0.40:
            score = score * 0.95  # 5% reduction
        
        return min(1.0, max(0.0, score))
    
    def _classify_risk(self, score: float) -> Tuple[str, str]:
        """Classify score into verdict and risk level"""
        if score >= self.thresholds['critical']:
            return 'CRITICAL_PHISHING', 'CRITICAL'
        elif score >= self.thresholds['high']:
            return 'PHISHING', 'HIGH'
        elif score >= self.thresholds['medium']:
            return 'SUSPICIOUS', 'MEDIUM'
        elif score >= self.thresholds['low']:
            return 'LOW_RISK', 'LOW'
        elif score >= self.thresholds['negligible']:
            return 'NEGLIGIBLE', 'NEGLIGIBLE'
        else:
            return 'LEGITIMATE', 'NEGLIGIBLE'
    
    def _generate_recommendation(self, score: float, risk_level: str, signal_count: int) -> str:
        """Generate user-friendly recommendation based on risk level"""
        recommendations = {
            'CRITICAL': (
                "🚨 CRITICAL: DO NOT interact with this content. "
                "This shows strong indicators of a sophisticated phishing attack. "
                "Delete immediately and report to security team."
            ),
            'HIGH': (
                "⚠️ HIGH RISK: This is likely a phishing attempt. "
                "Do not click any links, download attachments, or provide personal information. "
                "Verify the sender through official channels."
            ),
            'MEDIUM': (
                "⚠️ SUSPICIOUS: Multiple suspicious indicators detected. "
                "Exercise extreme caution. Verify the authenticity before proceeding. "
                "Contact the organization directly using known contact information."
            ),
            'LOW': (
                "🔍 LOW RISK: Some minor suspicious elements found. "
                "Review carefully but may be legitimate. "
                "Check for any unusual requests before proceeding."
            ),
            'NEGLIGIBLE': (
                "✅ NEGLIGIBLE RISK: No significant threats detected. "
                "Normal interaction should be safe, but always remain vigilant."
            )
        }
        
        # Default recommendation if risk level not found
        return recommendations.get(risk_level, (
            "Unable to determine risk level. Please exercise caution "
            "and verify through independent means."
        ))
    
    def _clean_evidence(self, evidence: List[str]) -> List[str]:
        """Clean and deduplicate evidence list"""
        # Remove duplicates while preserving order
        seen = set()
        unique_evidence = []
        
        for item in evidence:
            if item not in seen and item.strip():
                seen.add(item)
                unique_evidence.append(item)
        
        return unique_evidence
    
    def _get_individual_risk_levels(self,
                                   text_result: Optional[Dict],
                                   url_result: Optional[Dict],
                                   stego_prob: Optional[float]) -> Dict[str, str]:
        """Extract individual risk levels from each modality"""
        risks = {}
        
        if text_result:
            risks['text'] = text_result.get('risk_level', 'UNKNOWN')
        
        if url_result:
            risks['url'] = url_result.get('risk_level', 'UNKNOWN')
        
        if stego_prob is not None:
            if stego_prob > 0.65:
                risks['image'] = 'HIGH'
            elif stego_prob > 0.45:
                risks['image'] = 'MEDIUM'
            elif stego_prob > 0.25:
                risks['image'] = 'LOW'
            else:
                risks['image'] = 'NEGLIGIBLE'
        
        return risks
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create result for no input case"""
        return {
            'score': 0.0,
            'verdict': 'NO_DATA',
            'risk_level': 'UNKNOWN',
            'confidence': 0.0,
            'evidence': ['No input data provided for analysis'],
            'recommendation': 'Please provide text, URL, or image for analysis',
            'signals_processed': 0,
            'conflict_detected': False,
            'modality_scores': {},
            'individual_risks': {},
            'weights_used': {}
        }
    
    def update_weights(self, text_weight: float = None, 
                      url_weight: float = None, 
                      image_weight: float = None) -> None:
        """Update fusion weights dynamically"""
        if text_weight is not None:
            self.weights['text'] = max(0.0, min(1.0, text_weight))
        if url_weight is not None:
            self.weights['url'] = max(0.0, min(1.0, url_weight))
        if image_weight is not None:
            self.weights['image'] = max(0.0, min(1.0, image_weight))
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total


class EnsembleFusionEngine(PhishingFusionEngine):
    """
    Extended fusion engine with ensemble methods for improved accuracy
    """
    
    def __init__(self):
        super().__init__()
        self.ensemble_methods = {
            'weighted': self._weighted_fusion,
            'majority': self._majority_vote,
            'conservative': self._conservative_fusion
        }
    
    def fuse_with_ensemble(self,
                          text_result: Optional[Dict] = None,
                          url_result: Optional[Dict] = None,
                          stego_prob: Optional[float] = None,
                          method: str = 'weighted') -> Dict[str, Any]:
        """
        Fuse results using specified ensemble method
        
        Args:
            text_result: Text analysis result
            url_result: URL analysis result
            stego_prob: Steganography probability
            method: Fusion method ('weighted', 'majority', 'conservative')
        """
        if method in self.ensemble_methods:
            return self.ensemble_methods[method](text_result, url_result, stego_prob)
        else:
            return self.fuse(text_result, url_result, stego_prob)
    
    def _majority_vote(self,
                      text_result: Optional[Dict],
                      url_result: Optional[Dict],
                      stego_prob: Optional[float]) -> Dict[str, Any]:
        """Majority vote fusion"""
        votes = []
        
        if text_result:
            votes.append('phishing' if text_result.get('probability', 0) > 0.5 else 'legitimate')
        if url_result:
            votes.append('phishing' if url_result.get('probability', 0) > 0.5 else 'legitimate')
        if stego_prob is not None:
            votes.append('phishing' if stego_prob > 0.5 else 'legitimate')
        
        if not votes:
            return self._create_empty_result()
        
        phishing_votes = votes.count('phishing')
        total_votes = len(votes)
        
        if phishing_votes > total_votes / 2:
            score = 0.75 + (phishing_votes - total_votes/2) * 0.1
        else:
            score = 0.25 - (total_votes/2 - phishing_votes) * 0.1
        
        return {
            'score': min(1.0, max(0.0, score)),
            'verdict': 'PHISHING' if phishing_votes > total_votes / 2 else 'LEGITIMATE',
            'risk_level': 'HIGH' if phishing_votes > total_votes / 2 else 'LOW',
            'confidence': phishing_votes / total_votes,
            'evidence': [f"Majority vote: {phishing_votes}/{total_votes} signals indicate phishing"],
            'recommendation': self._generate_recommendation(score, 'HIGH' if phishing_votes > total_votes / 2 else 'LOW', total_votes),
            'signals_processed': total_votes,
            'conflict_detected': False,
            'modality_scores': {},
            'individual_risks': {},
            'weights_used': {}
        }
    
    def _conservative_fusion(self,
                           text_result: Optional[Dict],
                           url_result: Optional[Dict],
                           stego_prob: Optional[float]) -> Dict[str, Any]:
        """Conservative fusion - favors safety (higher risk)"""
        result = self.fuse(text_result, url_result, stego_prob)
        
        # Apply safety boost
        result['score'] = min(1.0, result['score'] * 1.15)
        
        # Reclassify risk
        verdict, risk_level = self._classify_risk(result['score'])
        result['verdict'] = verdict
        result['risk_level'] = risk_level
        
        return result


# Create singleton instances
fusion_engine = PhishingFusionEngine()
ensemble_engine = EnsembleFusionEngine()

# Export the main fusion function for backward compatibility
def fusion(Pv=None, Pt=None, Pu=None, Pstego=None):
    """
    Legacy fusion function for backward compatibility
    
    Args:
        Pv: Visual model probability (mapped to text for compatibility)
        Pt: Text model probability
        Pu: URL model probability
        Pstego: Steganography probability
    """
    text_result = {'probability': Pt} if Pt is not None else None
    url_result = {'probability': Pu} if Pu is not None else None
    
    result = fusion_engine.fuse(
        text_result=text_result,
        url_result=url_result,
        stego_prob=Pstego
    )
    
    return {
        'score': result['score'],
        'label': result['verdict'],
        'explanation': result['evidence'],
        'risk_level': result['risk_level'],
        'confidence': result['confidence']
    }


# =========================================================
# TEST FUNCTION
# =========================================================

def test_fusion_engine():
    """Test the fusion engine with sample data"""
    print("\n" + "="*60)
    print("🧪 TESTING FUSION ENGINE")
    print("="*60)
    
    # Test Case 1: All signals indicate phishing
    print("\n📊 Test Case 1: Clear Phishing")
    text_result = {'probability': 0.85, 'risk_level': 'HIGH', 'reasons': ['Urgent language', 'Account suspension']}
    url_result = {'probability': 0.78, 'risk_level': 'HIGH', 'reasons': ['Suspicious domain', 'No HTTPS']}
    stego_prob = 0.25
    
    result = fusion_engine.fuse(text_result, url_result, stego_prob)
    print(f"  Score: {result['score']:.3f}")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Risk: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Evidence: {len(result['evidence'])} items")
    
    # Test Case 2: Conflicting signals
    print("\n📊 Test Case 2: Conflicting Signals")
    text_result = {'probability': 0.82, 'risk_level': 'HIGH', 'reasons': ['Phishing keywords']}
    url_result = {'probability': 0.15, 'risk_level': 'LOW', 'reasons': ['Trusted domain']}
    stego_prob = 0.30
    
    result = fusion_engine.fuse(text_result, url_result, stego_prob)
    print(f"  Score: {result['score']:.3f}")
    print(f"  Conflict Detected: {result['conflict_detected']}")
    print(f"  Verdict: {result['verdict']}")
    
    # Test Case 3: Legitimate content
    print("\n📊 Test Case 3: Legitimate")
    text_result = {'probability': 0.12, 'risk_level': 'LOW', 'reasons': ['Normal communication']}
    url_result = {'probability': 0.08, 'risk_level': 'LOW', 'reasons': ['HTTPS', 'Trusted domain']}
    stego_prob = 0.10
    
    result = fusion_engine.fuse(text_result, url_result, stego_prob)
    print(f"  Score: {result['score']:.3f}")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Recommendation: {result['recommendation'][:50]}...")
    
    # Test Case 4: Single modality
    print("\n📊 Test Case 4: Single Modality")
    result = fusion_engine.fuse(text_result={'probability': 0.67, 'risk_level': 'MEDIUM'})
    print(f"  Score: {result['score']:.3f}")
    print(f"  Signals: {result['signals_processed']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    
    print("\n" + "="*60)
    print("✅ FUSION ENGINE TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_fusion_engine()