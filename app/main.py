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
from contextlib import AbstractAsyncContextManager

#our modules imported
from models.classifier import IntentClassifier
from api.routes import router, init_classifier

#logging config
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#global variable for the classifier
app_classifier = None

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
            logger.info("No model found. Begin training new...") #if there ain't one, train a new one

            #check for data archive (kept it simple do to time constraints)
            data_files = {
                "data/intent_dataset.json",
                "data/intent_dataset.csv"
            }

            data_file = None
            for file_path in data_files:
                if os.path.exists(file_path):
                    data_file =file_path
                    break

            if data_file:
                logger.info(f"Using data archive: {data_file}")
                metrics = app_classifier.train(data_file)
                app_classifier.save_model()
                logger.info(f"Model has been trained! Accuracy of {metrics['test_accuracy']:.4f}")
            else:
                logger.warning("No data found for training!")
                logger.info("Creating sample data for demonstration...")
                create_sample_data()  #thought it would be a nice extra and it ain't that hard to do
                metrics = app_classifier.train("intent_dataset.json")
                app_classifier.save_model()
                logger.info(f"Model trained with sample data! Accuracy {metrics['test_accuracy']:.4f}")
        
        #init classifier on routing
        init_classifier(app_classifier)
        logger.info("API is ready!")
    
    except Exception as e:
        logger.error(f"Error on initialization: {str(e)}")
        sys.exit(1)

    yield

    #shutdown
    logger.info("Shutting down API!")


#nice little extra
def create_sample_data():
    """
    Create a sample dataset for demonstration purposes only
    """
    import json

    sample_data = [
        # search_product
        {"text": "whey protein", "intent": "search_product"},
        {"text": "procurar creatina", "intent": "search_product"},
        {"text": "ver proteínas", "intent": "search_product"},
        {"text": "mostrar suplementos", "intent": "search_product"},
        {"text": "barras proteicas", "intent": "search_product"},
        {"text": "omega 3", "intent": "search_product"},
        {"text": "vitaminas", "intent": "search_product"},
        {"text": "bcaa", "intent": "search_product"},
        {"text": "glutamina", "intent": "search_product"},
        {"text": "casein", "intent": "search_product"},
        
        # product_info
        {"text": "como tomar creatina", "intent": "product_info"},
        {"text": "benefícios da whey", "intent": "product_info"},
        {"text": "diferença entre whey isolada e concentrada", "intent": "product_info"},
        {"text": "para que serve omega 3", "intent": "product_info"},
        {"text": "quando tomar bcaa", "intent": "product_info"},
        {"text": "efeitos colaterais glutamina", "intent": "product_info"},
        {"text": "informações sobre este produto", "intent": "product_info"},
        {"text": "composição nutricional", "intent": "product_info"},
        {"text": "modo de usar", "intent": "product_info"},
        {"text": "ingredientes", "intent": "product_info"},
        
        # add_to_cart
        {"text": "adicionar ao carrinho", "intent": "add_to_cart"},
        {"text": "comprar este produto", "intent": "add_to_cart"},
        {"text": "quero 2 unidades", "intent": "add_to_cart"},
        {"text": "por no cesto", "intent": "add_to_cart"},
        {"text": "adicionar 1kg", "intent": "add_to_cart"},
        {"text": "levar 3 potes", "intent": "add_to_cart"},
        {"text": "quero comprar", "intent": "add_to_cart"},
        {"text": "adicionar carrinho", "intent": "add_to_cart"},
        {"text": "colocar no carrinho", "intent": "add_to_cart"},
        {"text": "pôr na cesta", "intent": "add_to_cart"},
        
        # go_to_shopping_cart
        {"text": "ver carrinho", "intent": "go_to_shopping_cart"},
        {"text": "mostrar cesto", "intent": "go_to_shopping_cart"},
        {"text": "ir para carrinho", "intent": "go_to_shopping_cart"},
        {"text": "meu carrinho", "intent": "go_to_shopping_cart"},
        {"text": "ver cesta", "intent": "go_to_shopping_cart"},
        {"text": "carrinho de compras", "intent": "go_to_shopping_cart"},
        {"text": "produtos no carrinho", "intent": "go_to_shopping_cart"},
        {"text": "abrir carrinho", "intent": "go_to_shopping_cart"},
        {"text": "visualizar cesto", "intent": "go_to_shopping_cart"},
        {"text": "cesta de compras", "intent": "go_to_shopping_cart"},
        
        # confirm_order
        {"text": "finalizar compra", "intent": "confirm_order"},
        {"text": "confirmar pedido", "intent": "confirm_order"},
        {"text": "pagar agora", "intent": "confirm_order"},
        {"text": "checkout", "intent": "confirm_order"},
        {"text": "concluir encomenda", "intent": "confirm_order"},
        {"text": "confirmar encomenda", "intent": "confirm_order"},
        {"text": "efetuar pagamento", "intent": "confirm_order"},
        {"text": "finalizar pedido", "intent": "confirm_order"},
        {"text": "completar compra", "intent": "confirm_order"},
        {"text": "processar pedido", "intent": "confirm_order"},
        
        # unknown_intention
        {"text": "olá", "intent": "unknown_intention"},
        {"text": "obrigado", "intent": "unknown_intention"},
        {"text": "ajuda", "intent": "unknown_intention"},
        {"text": "suporte", "intent": "unknown_intention"},
        {"text": "não sei", "intent": "unknown_intention"},
        {"text": "talvez", "intent": "unknown_intention"},
        {"text": "pode ser", "intent": "unknown_intention"},
        {"text": "horários", "intent": "unknown_intention"},
        {"text": "localização", "intent": "unknown_intention"},
        {"text": "contacto", "intent": "unknown_intention"},
    ]

    with open("data/intent_dataset_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📝 Criados {len(sample_data)} exemplos de dados")

#creating FastAPI app
app = FastAPI(
    title="Intent Classification API",
    description="REST API for intent classification of portuguese text for e-commerce",
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
    Main API route
    """
    return {
        "message": "Intent Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "classify": "/api/v1/classify"
    }

#global exception handler
@app.exception_handler(Exception)
async def global_exception_hnadler(request, exc):
    """
    Global handler for unseen exceptions
    """
    logger.error(f"Error not treated: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An error on the server has ocurred",
            "timestamp": str(logger.handlers[0].formatter.formatTime(logging.LogRecord("",0, "", 0, "", (), None)))
        }
    )

if __name__ == "__main__":
    """
    App running
    """
    print("Starting FastAPI Server...")
    print("Documentation available on http://127.0.0.1:8000/docs")
    print("API Testing on http://127.0.0.1:8000/api/v1/classify")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )