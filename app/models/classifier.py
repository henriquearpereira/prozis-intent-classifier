"""
Este módulo contém a lógica para treinar e salvar um modelo de classificação de intenções.
Ele inclui funções para:
- Carregar e pré-processar dados
- Treinar um modelo de classificação
- Avaliar o modelo
- Salvar o modelo treinado
- Carregar um modelo treinado
- Fazer previsões com um modelo carregado
"""

import json
import pandas as pd
import numpy as np
import re
import os
import unicodedata
import logging
from typing import Tuple, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentClassifier:
    """
    Classe para treinar e salvar um modelo de classificação de intenções.
    Suporte para 6 intenções:
    - search_product: pesquisa de produtos
    - product_info: informações sobre um produto
    - add_to_cart: adicionar um produto ao carrinho
    - go_to_shopping_cart: ir para o carrinho de compras
    - confirm_order: confirmar pedido
    - unknown_intent: intenção desconhecida
    """
    def __init__(self, model_path: str = "app/models/"):
        """
        init the classifier 

        args:
            model_path: path to save the trained model
        """
        self.model_path = model_path
        self.vectorizer_file = os.path.join(model_path, "vectorizer.pkl")
        self.model_file = os.path.join(model_path, "intent_classifier.pkl")

        #init components 
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3), #better context
            lowercase=True,
            strip_accents=None,  #accents handled manually
            token_pattern=r"(?u)\b\w+\b",  #standard token pattern
            min_df=1,  #keep all words (small dataset)
            max_df=0.95  #remove very common words
        )

        self.model = LogisticRegression(
            C=1.0,  #regularization strength
            random_state=42,
            max_iter=1000,
            class_weight="balanced"  #handle class imbalance
        )

        self.is_trained = False

        self.intent_labels = {
            "search_product": 0,
            "product_info": 1,
            "add_to_cart": 2,
            "go_to_shopping_cart": 3,
            "confirm_order": 4,
            "unknown_intent": 5
        }

        # Portuguese stopwords (basic set)
        self.portuguese_stopwords = {
            'a', 'ao', 'aos', 'as', 'da', 'das', 'de', 'do', 'dos', 'e', 'em', 
            'na', 'nas', 'no', 'nos', 'o', 'os', 'para', 'por', 'que', 'se', 
            'um', 'uma', 'uns', 'umas', 'com', 'como', 'mais', 'mas', 'ou', 
            'ser', 'ter', 'esta', 'este', 'isso', 'sua', 'seu', 'dela', 'dele'
        }


    def preprocess_text(self, text: str) -> str:
        """
        Preprocess portuguesetext to remove accents, stopwords, and special characters

        args:
            text: input text to preprocess

        returns:
            preprocessed text
        """

        if not text or not isinstance(text,str):
            return ""
        
        #to lowercase
        text = text.lower()

        #remove accents and diacritics
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if not unicodedata.category(c) == "Mn")

        #clean up punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        #remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        #remove standalone numbers (keep alphanumeric mentions like "1kg" or "2x")
        text = re.sub(r'\b\d+\b', '', text)

        #remove portuguese stopwords
        text = " ".join(word for word in text.split() if word not in self.portuguese_stopwords)

        # Remove very short words (1 char) that aren't meaningful
        words = text.split()
        words = [word for word in words if len(word) > 1 or word in ['a', 'o', 'e']]
        
        return ' '.join(words)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Loading the data either from JSON or CSV File

        args:
            data_path: path to the dataset file
        
        returns:
            dataframe with text and label columns
        
        """
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path, encoding= "utf-8")
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            #standardize column names
            if 'label' in df.columns:
                df = df.rename(columns={'label': 'intent'})
            elif 'intent' not in df.columns:
                raise ValueError("Dataset must have 'label' or 'intent' column")
            
            #validation intents
            unknown_intents = set(df['intent'].unique()) - set(self.intent_labels)
            if unknown_intents:
                logger.warning(f"Unknown intents found: {unknown_intents}")

            logger.info(f"Loaded {len(df)} samples from {data_path}")
            logger.info(f"Intent distribution:\n {df['intent'].value_counts()}")

            return df
        
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise

    def train(self, data_path: str, test_size: float = 0.2) -> dict:
        """
        Train the classifier on the provided dataset

        args:
            - data_path: path to the training dataset
            - test_size: fraction of dtat to use for testing
        
        returns:
            dict with training metrics
        """
        try: 
            #load data
            df = self.load_data(data_path)

            #preprocess texts
            df['processed_text'] = df['text'].apply(self.preprocess_text)

            #removal of empty texts 
            df = df[df['processed_text'].str.len() > 0]

            if len(df) == 0:
                raise ValueError("No valid texts after preprocessing")
            
            #preparing features and labels
            X = df['processed_text'].values
            y = df['intent'].values

            #split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

            #vectorize test
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)

            #train model
            self.model.fit(X_train_tfidf, y_train)

            #evaluation on testing set
            y_pred = self.model.predict(X_test_tfidf)
            y_pred_proba = self.model.predict_proba(X_test_tfidf)

            #calculate metrics
            test_accuracy = self.model.score(X_test_tfidf, y_test)

            #cross val on training set
            cv_scores = cross_val_score(
                self.model, X_train_tfidf, y_train, cv=3, scoring='accuracy'
            )

            #generate classification report 
            report = classification_report(y_test, y_pred, output_dict=True)

            #log results
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
            
            # Feature importance (top features per class)
            feature_names = self.vectorizer.get_feature_names_out()
            self._log_feature_importance(feature_names)
            
            self.is_trained = True
            
            # Return metrics
            metrics = {
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': report,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _log_feature_importance(self, feature_names: np.ndarray, top_n: int=5):
        """
        Logging top features for each class
        """
        try: 
            for i, class_name in enumerate(self.model.classes_):
                coef = self.model.coef_[i] #get the coefficients
                #now get top positive features
                top_indices = np.argsort(coef)[-top_n:][::-1]
                top_features = [feature_names[idx] for idx in top_indices]
                top_scores = [coef[idx] for idx in top_indices]

                logger.info(f"Top features for '{class_name}':")
                for feature, score in zip(top_features, top_scores):
                    logger.info(f"  {feature}: {score:.4f}")
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")

    def predict (self, text: str)  -> Tuple[str,float]:
        """
        Prediction of intent for a single text
        
        args:
            text: input user text for classification
            
        returns:
            tuple of (predicted_intent, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained still. Call train() first please")
        
        if not text or not isinstance(text, str):
            return "unknown_intention", 0.0
        
        #preprocess
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return "unknown_intention", 0.0
        
        #vectorization
        text_tfidf = self.vectorizer.transform([processed_text])

        #prediction
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]

        #get confidence (max proba)
        confidence = float(np.max(probabilities))

        #log prediction details
        logger.debug(f"Input: '{text}' -> Processed: '{processed_text}' ")
        logger.debug(f"Prediction: {prediction} (w/ confidence: {confidence:.4f})")

        return prediction, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str,float]]:
        """
        Precition of intents for multiple texts

        args:
            texts: list of input texts

        returns:
            list of (predicted_intent, confidence_score) tuples
        """

        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first please")
        
        results = []
        for text in texts:
            try:
                intent, confidence  = self.predict(text)
                results.append((intent,confidence))
            except Exception as e:
                logger.warning(f"Error predicting text '{text}': {e}")
                results.append(("unknown_intention", 0.0))

        return results
    
    def save_model(self):
        """
        Save the trained model and vectorizer
        """
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            if not self.is_trained:
                raise ValueError("No trained model to save")
            
            joblib.dump(self.vectorizer, self.vectorizer_file)
            joblib.dump(self.model, self.model_file)
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self) -> bool:
        """
        Load a previously trained model
        
        Returns:
            True if model loaded successfully
            False otherwise
        """
        try:
            if not os.path.exists(self.vectorizer_file) or not os.path.exists(self.model_file):
                logger.info("No saved model found")
                return False
            
            self.vectorizer = joblib.load(self.vectorizer_file)
            self.model = joblib.load(self.model_file)
            self.is_trained = True
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "supported_intents": self.intent_labels,
            "vectorizer_features": self.vectorizer.get_feature_names_out().shape[0] if hasattr(self.vectorizer, 'get_feature_names_out') else 0,
            "model_type": type(self.model).__name__,
            "model_params": self.model.get_params()
        }



# #Example use
# if __name__ == '__main__':
#     #init classifier
#     classifier = IntentClassifier()
#     sample_data= "data\intent_dataset.json"
#     try:
#         print("Training the model...")
#         metrics = classifier.train(sample_data)
#         print(f"Training completed w/ accuracy: {metrics['test_accuracy']:.4f}")

#         #test predictions
#         test_texts = [
#             "whey protein",
#             "como usar creatina",
#             "adicionar ao carrinho",
#             "ver meu carrinho",
#             "finalizar compra",
#             "ola"
#         ]
        
#         print("\nTesting predictions:")
#         for text in test_texts:
#             intent, confidence = classifier.predict(text)
#             print(f"'{text}' -> {intent} (confidence: {confidence:.4f})")
        
#         # Save model
#         classifier.save_model()
#         print("Model saved successfully")
        
#     except Exception as e:
#         print(f"Error: {e}")

