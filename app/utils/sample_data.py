import os 
import logging
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_sample_data():
    """
    Create a comprehensive sample dataset for demonstration purposes
    Covers all 6 required intents with Portuguese examples
    """

    sample_data = [
        # search_product - Enhanced with more variations
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
        {"text": "whey isolate", "intent": "search_product"},
        {"text": "ver multivitaminicos", "intent": "search_product"},
        {"text": "barra de amendoim", "intent": "search_product"},
        {"text": "procurar aveia", "intent": "search_product"},
        {"text": "suplementos fitness", "intent": "search_product"},
        
        # product_info - Enhanced with more question patterns
        {"text": "como tomar creatina", "intent": "product_info"},
        {"text": "benefícios da whey", "intent": "product_info"},
        {"text": "diferença entre whey concentrada e isolada", "intent": "product_info"},
        {"text": "para que serve omega 3", "intent": "product_info"},
        {"text": "quando tomar bcaa", "intent": "product_info"},
        {"text": "efeitos colaterais glutamina", "intent": "product_info"},
        {"text": "informações sobre este produto", "intent": "product_info"},
        {"text": "composição nutricional", "intent": "product_info"},
        {"text": "modo de usar", "intent": "product_info"},
        {"text": "ingredientes", "intent": "product_info"},
        {"text": "qual a diferença entre whey concentrada e isolada", "intent": "product_info"},
        {"text": "benefícios omega 3", "intent": "product_info"},
        {"text": "como tomar creatina monohidrato", "intent": "product_info"},
        {"text": "dosagem recomendada", "intent": "product_info"},
        {"text": "contraindicações", "intent": "product_info"},
        
        # add_to_cart - Enhanced with quantity variations
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
        {"text": "adicionar 1kg de aveia ao carrinho", "intent": "add_to_cart"},
        {"text": "quero comprar 2 caixas de barras", "intent": "add_to_cart"},
        {"text": "põe a creatina no cesto", "intent": "add_to_cart"},
        {"text": "comprar 500g", "intent": "add_to_cart"},
        {"text": "levar este produto", "intent": "add_to_cart"},
        
        # go_to_shopping_cart - Enhanced variations
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
        {"text": "mostrar carrinho", "intent": "go_to_shopping_cart"},
        {"text": "ir para o cesto", "intent": "go_to_shopping_cart"},
        {"text": "quero finalizar", "intent": "go_to_shopping_cart"},
        {"text": "ver itens no carrinho", "intent": "go_to_shopping_cart"},
        {"text": "acessar carrinho", "intent": "go_to_shopping_cart"},
        
        # confirm_order - Enhanced with payment terms
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
        {"text": "confirmar encomenda", "intent": "confirm_order"},
        {"text": "pagar agora", "intent": "confirm_order"},
        {"text": "finalizar compra", "intent": "confirm_order"},
        {"text": "proceder ao pagamento", "intent": "confirm_order"},
        {"text": "concluir compra", "intent": "confirm_order"},
        
        # unknown_intention - Enhanced with edge cases
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
        {"text": "obrigado", "intent": "unknown_intention"},
        {"text": "preciso de ajuda", "intent": "unknown_intention"},
        {"text": "falar com suporte", "intent": "unknown_intention"},
        {"text": "promoções", "intent": "unknown_intention"},
        {"text": "bom dia", "intent": "unknown_intention"},
        {"text": "boa tarde", "intent": "unknown_intention"},
        {"text": "até logo", "intent": "unknown_intention"},
        {"text": "como está", "intent": "unknown_intention"},
        {"text": "não entendo", "intent": "unknown_intention"},
        {"text": "desculpe", "intent": "unknown_intention"},
    ]

    with open("data/intent_dataset_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📝 Created {len(sample_data)} sample data examples covering all 6 intents")
    
    # Log intent distribution
    intent_counts = {}
    for item in sample_data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    logger.info("Intent distribution in sample data:")
    for intent, count in intent_counts.items():
        logger.info(f"  {intent}: {count} examples")