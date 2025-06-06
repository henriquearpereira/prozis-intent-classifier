"""
API Routes
Toma conta dos endpoints REST para classificação de texto
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime

#import our classifier 
from models.classifier import IntentClassifier

#logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#router init
router=APIRouter()

#global classifier instance 
classifier: Optional[IntentClassifier] = None

#request/response models
class TextRequest(BaseModel):
    """
    Request the model for single text classification
    """
    text: str = Field(..., min_length=1, max_length=1000, description="Text to be classified")

    @field_validator('text')
    def validate_text(cls,v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()
    
class BatchTextRequest(BaseModel):
    """
    Request model for batch text classification
    """
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")

    @field_validator('texts')
    def validate_texts(cls,v):
        if not v:
             raise ValueError('Text list cannot be empty')
        
        cleaned_texts=[]
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty')
            if len(text) > 1000:
                raise ValueError(f'Text at index {i} is too long (max 1000 characters)')
            cleaned_texts.append(text.strip())

        return cleaned_texts
    
class IntentResponse(BaseModel):
    """
    Response model for single text classification
    """
    intent: str = Field(..., description="Predicted intent")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchIntentResponse(BaseModel):
    """
    Response model for batch text classification
    """
    intent: List[IntentResponse]= Field(..., description="List of classification results")
    total_processed: int = Field(..., description="Total number of tects processed")
    avg_processing_time_ms: float = Field(..., description="Average processing time per text")

class ModelInfoResponse(BaseModel):
    """
    Response model for model info
    """
    status: str = Field(..., description="Model Status")
    supported_intents: List[str] = Field(..., description="List of supported intents")
    model_type: str = Field(..., description= "Type of ML model used")
    vectorizer_features: int = Field(..., description="Number of features in vectorizer")
    last_trained: Optional[str] = Field(None, description="Last training timestamp")

class HealthResponse(BaseModel):
    """
    Response model for health check
    """
    status: str = Field(..., description="API status")
    model_ready: bool = Field(..., description="whether model is ready or not for prediction")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    supported_intents: List[str] = Field(..., description="List of supported intents")

class ErrorResponse(BaseModel):
    """
    Response model for errors
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(..., description="Detailed error info")
    timestamp: str = Field(..., description="Error timestamp")

#global variables for health tracking
start_time=time.time()

#get classifier instance
def get_classifier() -> IntentClassifier:
    """
    dependency to get the classifier instance
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not initialized. Please check server startup logs."
        )
    if not classifier.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train the model first."
        )
    return classifier

#classification endpoint
@router.post(
    "/classify",
    response_model=IntentResponse,
    summary="Classify text intent",
    description="Classify a single text input into one of the predefined intents",
    responses={
        200: {"description": "successfully classified text"},
        400: {"description": "invalid input"},
        503: {"description": "service unavailable - model is not ready"}
    }
)

async def classify_text(request: TextRequest,
    classifier_instance: IntentClassifier = Depends(get_classifier)) -> IntentResponse:
    """
    Classifying a single text into one of the predefined intents
    the text to classify can be between 1 and 1000 characters
    returns the predicted intent w/ confidence score and its processing time
    """
    start_time_ms =time.time()*1000

    try:
        #classify the text
        intent, confidence = classifier_instance.predict(request.text)
        #calculating the processing time
        processing_time = (time.time()*1000) - start_time_ms
        #log the prediction
        logger.info(f"Classified '{request.text[:50]} ...' -> {intent} (confidence: {confidence:.4f})")

        return IntentResponse(
            intent=intent,
            confidence_score=round(confidence,4),
            processing_time_ms=round(processing_time, 2)
        )
    
    except Exception as e:
        logger.error(f"Classification error for text '{request.text}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

# Batch classification endpoint
@router.post(
    "/classify/batch",
    response_model=BatchIntentResponse,
    summary="Classify multiple texts",
    description="Classify multiple text inputs at once (max 100 texts)",
    responses={
        200: {"description": "Successfully classified all texts"},
        400: {"description": "Invalid input"},
        503: {"description": "Service unavailable - model not ready"}
    }
)
async def classify_batch(
    request: BatchTextRequest,
    classifier_instance: IntentClassifier = Depends(get_classifier)) -> BatchIntentResponse:
    """
    Classify multiple texts into predefined intents
    
    List of texts to classify (1-100 texts, each 1-1000 characters)
    
    Returns a list of predictions with individual processing times.
    """
    overall_start = time.time() * 1000
    results = []
    
    try:
        for i, text in enumerate(request.texts):
            start_time_ms = time.time() * 1000
            
            try:
                intent, confidence = classifier_instance.predict(text)
                processing_time = (time.time() * 1000) - start_time_ms
                
                results.append(IntentResponse(
                    intent=intent,
                    confidence_score=round(confidence, 4),
                    processing_time_ms=round(processing_time, 2)
                ))
                
            except Exception as e:
                logger.error(f"Error classifying text {i}: {str(e)}")
                # Add error result but continue processing others
                results.append(IntentResponse(
                    intent="unknown_intention",
                    confidence_score=0.0,
                    processing_time_ms=0.0
                ))
        
        # Calculate average processing time
        total_time = (time.time() * 1000) - overall_start
        avg_time = total_time / len(request.texts) if request.texts else 0
        
        logger.info(f"Batch classified {len(request.texts)} texts in {total_time:.2f}ms")
        
        return BatchIntentResponse(
            intent=results,
            total_processed=len(results),
            avg_processing_time_ms=round(avg_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch classification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch classification failed: {str(e)}"
        )

# Model information endpoint
@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get detailed information about the trained model"
)

async def get_model_info(
    classifier_instance: IntentClassifier = Depends(get_classifier)) -> ModelInfoResponse:
    """
    Get information about the current trained model
    
    Returns model status, supported intents, and technical details.
    """
    try:
        model_info = classifier_instance.get_model_info()
        
        return ModelInfoResponse(
            status=model_info.get("status", "unknown"),
            supported_intents=list(model_info.get("supported_intents", {}).keys()),
            model_type=model_info.get("model_type", "Unknown"),
            vectorizer_features=model_info.get("vectorizer_features", 0),
            last_trained=datetime.now().isoformat()  
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )

# Health check endpoint
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API and model health status"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns API status, model readiness, and uptime information.
    """
    try:
        uptime = time.time() - start_time
        model_ready = classifier is not None and classifier.is_trained if classifier else False
        
        supported_intents = []
        if classifier:
            supported_intents = getattr(classifier, 'intent_labels', [])
        
        return HealthResponse(
            status="healthy" if model_ready else "degraded",
            model_ready=model_ready,
            uptime_seconds=round(uptime, 2),
            supported_intents=supported_intents
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_ready=False,
            uptime_seconds=0.0,
            supported_intents=[]
        )

# Model retrain endpoint 
@router.post(
    "/model/retrain",
    summary="Retrain model",
    description="Retrain the model with new data (requires data file)",
    responses={
        200: {"description": "Model retrained successfully"},
        400: {"description": "Invalid request"},
        500: {"description": "Training failed"}
    }
)
async def retrain_model(
    background_tasks: BackgroundTasks,
    data_path: str = "data/intent_dataset.json") -> Dict[str, Any]:
    """
    Retrain the model with new data
    
    args:
    - **data_path**: Path to the training dataset, you can alter the file with more data or different data to retrain, the chosen one is still the json file provided but could any other 
    
    This endpoint triggers background retraining to avoid blocking the API.
    """
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not initialized"
        )
    
    def retrain_task():
        try:
            logger.info(f"Starting model retraining with data from {data_path}")
            metrics = classifier.train(data_path)
            classifier.save_model()
            logger.info(f"Model retrained successfully. Accuracy: {metrics['test_accuracy']:.4f}")
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
    
    # Add retraining task to background
    background_tasks.add_task(retrain_task)
    
    return {
        "message": "Model retraining started in background",
        "data_path": data_path,
        "timestamp": datetime.now().isoformat()
    }

# Statistics endpoint 
@router.get("/stats",
    summary="Get usage statistics",
    description="Get API usage statistics and model performance metrics")
async def get_statistics() -> Dict[str, Any]:
    """   
    Returns basic statistics about API usage and model performance.
    """
    try:
        uptime = time.time() - start_time
        
        stats = {
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "model_ready": classifier is not None and classifier.is_trained,
            "supported_intents_count": len(classifier.intent_labels) if classifier else 0,
            "api_version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        if classifier and classifier.is_trained:
            model_info = classifier.get_model_info()
            stats["model_features"] = model_info.get("vectorizer_features", 0)
            stats["model_type"] = model_info.get("model_type", "Unknown")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )

# Initialize classifier function (to be called from main.py)
def init_classifier(classifier_instance: IntentClassifier):
    """Initialize the global classifier instance"""
    global classifier
    classifier = classifier_instance
    logger.info("Classifier initialized in routes module")
    