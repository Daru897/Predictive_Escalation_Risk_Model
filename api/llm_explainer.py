from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import torch
from typing import List, Dict, Optional
import re
from functools import lru_cache
import time

# ----------------------------
# Setup logger
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "google/flan-t5-small"  # Lightweight model that runs locally
MAX_LENGTH = 80
NUM_BEAMS = 3
CACHE_SIZE = 100  # Cache size for explanations

# ----------------------------
# Model Loading with Optimization
# ----------------------------
@lru_cache(maxsize=1)
def load_model():
    """Load the model with caching and optimization"""
    logger.info("📥 Loading local LLM (flan-t5-small)...")
    start_time = time.time()
    
    try:
        # Load model with optimization settings
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Create pipeline with optimized settings
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.2f}s")
        logger.info(f"🔧 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        logger.info(f"🔧 Precision: {'float16' if torch.cuda.is_available() else 'float32'}")
        
        return generator
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Fallback to CPU-only loading
        try:
            logger.info("🔄 Trying CPU-only loading...")
            generator = pipeline(
                "text2text-generation",
                model=MODEL_NAME,
                device=-1,  # Force CPU
                torch_dtype=torch.float32
            )
            logger.info("✅ Model loaded on CPU")
            return generator
        except Exception as fallback_error:
            logger.error(f"❌ CPU loading also failed: {fallback_error}")
            return None

# ----------------------------
# Prompt Engineering Templates
# ----------------------------
PROMPT_TEMPLATES = {
    "default": (
        "Ticket {ticket_id} risk factors: {feature_summary}. "
        "Conversation: '{text_snippet}'. "
        "Write a short explanation (2 sentences) why this ticket might escalate."
    ),
    "technical": (
        "Based on these technical indicators: {feature_summary}. "
        "And this customer interaction: '{text_snippet}'. "
        "Provide a concise technical analysis of escalation risk."
    ),
    "business": (
        "Customer ticket {ticket_id} shows: {feature_summary}. "
        "From the conversation: '{text_snippet}'. "
        "Explain the business impact and escalation risk in simple terms."
    )
}

# ----------------------------
# Feature Processing
# ----------------------------
def format_features(top_features: List[Dict]) -> str:
    """Format features for the prompt in a readable way"""
    if not top_features:
        return "no specific risk factors identified"
    
    formatted = []
    for feature in top_features[:3]:  # Use top 3 features only
        feature_name = feature.get('feature', '').replace('_', ' ').title()
        impact = "increases" if feature.get('shap_value', 0) > 0 else "decreases"
        value = feature.get('feature_value')
        
        if value is not None:
            formatted.append(f"{feature_name} ({value}) {impact} risk")
        else:
            formatted.append(f"{feature_name} {impact} risk")
    
    return "; ".join(formatted)

def clean_text_snippet(text: str, max_length: int = 100) -> str:
    """Clean and truncate text snippet for the prompt"""
    if not text:
        return "no conversation text available"
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text.strip())  # Remove extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special chars
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."
    
    return text

# ----------------------------
# Explanation Generation with Caching
# ----------------------------
@lru_cache(maxsize=CACHE_SIZE)
def generate_explanation_cached(prompt: str) -> str:
    """Generate explanation with caching to avoid redundant computations"""
    generator = load_model()
    if generator is None:
        return "Explanation service unavailable."
    
    try:
        result = generator(
            prompt,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1
        )
        return result[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return "Could not generate explanation at this time."

def generate_explanation(ticket_id: int, top_features: List[Dict], text_snippet: str, 
                        style: str = "default") -> str:
    """
    Generate a human-readable explanation using local LLM with optimizations.
    """
    start_time = time.time()
    
    # Validate inputs
    if not top_features:
        logger.warning("⚠️ No features provided for explanation")
        return "Insufficient data to generate explanation."
    
    # Process inputs
    feature_summary = format_features(top_features)
    cleaned_text = clean_text_snippet(text_snippet)
    
    # Select prompt template
    template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES["default"])
    prompt = template.format(
        ticket_id=ticket_id,
        feature_summary=feature_summary,
        text_snippet=cleaned_text
    )
    
    logger.info(f"📝 Generated prompt: {prompt}")
    
    # Generate explanation
    try:
        explanation = generate_explanation_cached(prompt)
        
        # Post-process explanation
        explanation = post_process_explanation(explanation)
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Explanation generated in {generation_time:.2f}s: {explanation}")
        
        return explanation
        
    except Exception as e:
        logger.error(f"❌ Explanation generation failed: {e}")
        return fallback_explanation(feature_summary, cleaned_text)

def post_process_explanation(explanation: str) -> str:
    """Post-process the generated explanation for better readability"""
    # Capitalize first letter
    if explanation and explanation[0].islower():
        explanation = explanation[0].upper() + explanation[1:]
    
    # Ensure it ends with a period
    if explanation and not explanation.endswith(('.', '!', '?')):
        explanation += '.'
    
    # Remove redundant phrases
    redundant_phrases = [
        "the text says", "according to the conversation", 
        "based on the provided information", "the prompt states"
    ]
    
    for phrase in redundant_phrases:
        explanation = explanation.replace(phrase, "")
    
    # Clean up whitespace
    explanation = re.sub(r'\s+', ' ', explanation.strip())
    
    return explanation

def fallback_explanation(feature_summary: str, text_snippet: str) -> str:
    """Fallback explanation when LLM fails"""
    if "no specific risk factors" in feature_summary:
        return "This ticket shows typical patterns with no strong escalation indicators."
    
    if "increases risk" in feature_summary:
        return f"This ticket contains factors that may increase escalation risk: {feature_summary}."
    
    return "Based on the available data, this ticket requires further monitoring for potential escalation."

# ----------------------------
# Batch Processing
# ----------------------------
def batch_generate_explanations(tickets_data: List[Dict]) -> List[Dict]:
    """Generate explanations for multiple tickets efficiently"""
    results = []
    generator = load_model()
    
    if generator is None:
        return [{"ticket_id": data.get("ticket_id"), "explanation": "Service unavailable"} 
                for data in tickets_data]
    
    # Prepare all prompts first
    prompts = []
    for data in tickets_data:
        feature_summary = format_features(data.get("top_features", []))
        cleaned_text = clean_text_snippet(data.get("text_snippet", ""))
        prompt = PROMPT_TEMPLATES["default"].format(
            ticket_id=data.get("ticket_id", "unknown"),
            feature_summary=feature_summary,
            text_snippet=cleaned_text
        )
        prompts.append(prompt)
    
    # Generate in batch if supported, otherwise sequentially
    try:
        # Try batch generation
        batch_results = generator(
            prompts,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            num_return_sequences=1
        )
        
        for i, data in enumerate(tickets_data):
            explanation = batch_results[i]["generated_text"].strip()
            explanation = post_process_explanation(explanation)
            results.append({
                "ticket_id": data.get("ticket_id"),
                "explanation": explanation
            })
            
    except Exception:
        # Fallback to sequential generation
        logger.warning("⚠️ Batch generation failed, falling back to sequential")
        for data in tickets_data:
            explanation = generate_explanation(
                data.get("ticket_id", 0),
                data.get("top_features", []),
                data.get("text_snippet", "")
            )
            results.append({
                "ticket_id": data.get("ticket_id"),
                "explanation": explanation
            })
    
    return results

# ----------------------------
# Health Check
# ----------------------------
def check_model_health() -> Dict:
    """Check if the model is loaded and healthy"""
    generator = load_model()
    return {
        "loaded": generator is not None,
        "model": MODEL_NAME,
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "cache_size": CACHE_SIZE
    }

# Example usage
if __name__ == "__main__":
    # Test the explanation generator
    test_features = [
        {"feature": "ticket_age_hrs", "shap_value": 0.15, "feature_value": 48.5},
        {"feature": "num_transfers", "shap_value": 0.12, "feature_value": 3},
        {"feature": "sentiment", "shap_value": -0.08, "feature_value": -0.2}
    ]
    
    test_text = "Customer expressed frustration with delayed response and is asking to speak to a manager."
    
    explanation = generate_explanation(123, test_features, test_text)
    print(f"Generated explanation: {explanation}")
    
    # Health check
    health = check_model_health()
    print(f"Model health: {health}")