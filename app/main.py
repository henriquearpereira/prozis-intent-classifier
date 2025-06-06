"""
Aplicação principal para a API de classificação de intenções para o desafio da prozis
Init do modelo e configura os endpoints FastAPI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn 
import logging
import os
import sys
from contextlib import asynccontextmanager

#our modules imported
from models.classifier import IntentClassifier
from api.routes import router, init_classifier
from utils.sample_data import create_sample_data

#logging config
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#global variable for the classifier
app_classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manager of the app's life cycle
    Init the model on startup 
    cleans up the resources on shutdown
    """
    global app_classifier

    #@ startup
    logger.info("Booting up API of Intention Classification")

    try:
        #init classifier
        app_classifier= IntentClassifier(model_path="models/")
        
        #upload current model
        if app_classifier.load_model():
            logger.info("Model uploaded with success!")
        else:
            logger.info("No model found. Begin training new...") 

            # Enhanced dataset priority order
            data_files = [
                "data/enhanced_intent_dataset.json",  # Priority 1: Enhanced dataset
                "data/intent_dataset.json",           # Priority 2: Original dataset
                "data/intent_dataset.csv"             # Priority 3: CSV fallback
            ]

            data_file = None
            for file_path in data_files:
                if os.path.exists(file_path):
                    data_file = file_path
                    logger.info(f"Found dataset: {data_file}")
                    break

            if data_file:
                logger.info(f"Using data archive: {data_file}")
                metrics = app_classifier.train(data_file)
                app_classifier.save_model()
                logger.info(f"Model has been trained! Accuracy of {metrics['test_accuracy']:.4f}")
                
                # Log cross-validation metrics if available
                if 'cv_mean' in metrics:
                    logger.info(f"Cross-validation mean: {metrics['cv_mean']:.4f}")
                if 'cv_std' in metrics:
                    logger.info(f"Cross-validation std: {metrics['cv_std']:.4f}")
                    
            else:
                logger.warning("No data found for training!")
                logger.info("Creating sample data for demonstration...")
                
                # Ensure data directory exists
                os.makedirs("data", exist_ok=True)
                
                create_sample_data()
                metrics = app_classifier.train("data/intent_dataset_sample.json")
                app_classifier.save_model()
                logger.info(f"Model trained with sample data! Accuracy {metrics['test_accuracy']:.4f}")
        
        #init classifier on routing
        init_classifier(app_classifier)
        logger.info("API is ready!")
        logger.info(f"Supported intents: {list(app_classifier.intent_labels.keys())}")
    
    except Exception as e:
        logger.error(f"Error on initialization: {str(e)}")
        sys.exit(1)

    yield

    #shutdown
    logger.info("Shutting down API!")
    if app_classifier:
        logger.info("Cleaning up classifier resources...")

#creating FastAPI app
app = FastAPI(
    title="Intent Classification API - Prozis Challenge",
    description="REST API for intent classification of Portuguese text for e-commerce. Supports 6 predefined intents: search_product, product_info, add_to_cart, go_to_shopping_cart, confirm_order, unknown_intention",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

#CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #not in prod but this app is only for demonstration
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#routing
app.include_router(router, prefix="/api/v1", tags=["Classification"])

#main route
@app.get("/")
async def root():
    """
    Main API route - Shows available endpoints
    """
    return {
        "message": "Intent Classification API - Prozis Challenge",
        "version": "1.0.0",
        "supported_intents": [
            "search_product",
            "product_info", 
            "add_to_cart",
            "go_to_shopping_cart",
            "confirm_order",
            "unknown_intention"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/api/v1/health",
            "classify": "/api/v1/classify",
            "batch_classify": "/api/v1/classify/batch",
            "model_info": "/api/v1/model/info",
            "retrain": "/api/v1/model/retrain",
            "stats": "/api/v1/stats"
        }
    }

#global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global handler for unseen exceptions
    """
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An error on the server has occurred",
            "timestamp": str(logger.handlers[0].formatter.formatTime(logging.LogRecord("",0, "", 0, "", (), None)))
        }
    )

if __name__ == "__main__":
    """
    App running
    """
    print("🚀 Starting FastAPI Server for Prozis Intent Classification Challenge...")
    print("📚 Documentation available on http://127.0.0.1:8000/docs")
    print("🔍 API Testing on http://127.0.0.1:8000/api/v1/classify")
    print("💊 Health Check on http://127.0.0.1:8000/api/v1/health")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )