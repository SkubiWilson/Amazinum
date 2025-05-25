from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class SentimentDataLoader:
    def __init__(self, neg_path='rt-polarity.neg', pos_path='rt-polarity.pos'):
        self.neg_path = neg_path
        self.pos_path = pos_path
        self.texts_neg = []
        self.texts_neg = []
        self.texts_pos = []
        self.texts = []
        self.labels = []
        self.vectorizer = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.model = None

    def load_neg(self):
        with open(self.neg_path, "r", encoding='utf-8', errors='ignore') as f:
            self.texts_neg = f.read().splitlines()

    def load_pos(self):
        with open(self.pos_path, "r", encoding='utf-8', errors='ignore') as f:
            self.texts_pos = f.read().splitlines()

    def prepare_dataset(self):
        self.texts = self.texts_neg + self.texts_pos
        self.labels = [0]*len(self.texts_neg) + [1]*len(self.texts_pos)
        print(f'[INFO] Загальна кількість записів: {len(self.texts):,} ')

    def split_data(self, test_size=0.2, random_state=50):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.texts, self.labels, test_size=test_size, random_state=random_state)
        print(f'[INFO] Розділено: {len(self.X_train)} для тренування, {len(self.X_test)} для тестування')

    def vectorize (self):
        self.vectorizer = CountVectorizer()
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)
        return X_train_vec, X_test_vec

    def train_logistic_regresion(self, X_train_vec):
        self.model = LogisticRegression(max_iter=1200)
        self.model.fit(X_train_vec, self.y_train)
        print("[INFO] Logistic Regression fin")

    def evaluate_model(self, X_test_vec):
        y_pred = self.model.predict(X_test_vec)
        print("[Result] Оцінка моделі:")
        print(classification_report(self.y_test, y_pred, digits=4))

    def vectorize_tfidf(self):
        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        return X_train_tfidf, X_test_tfidf

    def train_naive_bayes(self, X_train_tfidf):
        self.model = MultinomialNB()
        self.model.fit(X_train_tfidf, self.y_train)
        print("[INFO] Naive Bayes fin")


class BertSentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', max_len=128, batch_size=16):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer.encode_plus(
                self.texts[idx],
                truncation=True,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    def train(self, texts, labels, epochs=3):
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

        train_dataset = self.SentimentDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_dataset = self.SentimentDataset(X_val, y_val, self.tokenizer, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()
        for epoch in range(epochs):
            print(f'\n[Epoch {epoch + 1}] ----------------------')
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        print("\n[RESULT] Оцінка моделі BERT:")
        print(classification_report(true_labels, predictions, digits=4))


loader = SentimentDataLoader()
loader.load_neg()
loader.load_pos()
loader.prepare_dataset()
loader.split_data()
X_train_vec, X_test_vec = loader.vectorize()
loader.train_logistic_regresion(X_train_vec)
loader.evaluate_model(X_test_vec)
X_train_tfidf, X_test_tfidf = loader.vectorize_tfidf()
loader.train_naive_bayes(X_train_tfidf)
loader.evaluate_model(X_test_tfidf)

texts, labels = loader.texts, loader.labels

bert_classifier = BertSentimentClassifier()
bert_classifier.train(texts, labels)



