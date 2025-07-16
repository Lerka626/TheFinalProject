import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import re
import string
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.corpus import stopwords

# Проверка на скачивание стоп-слов
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_data(df):
    """
    Предобрабатывает входной DataFrame:
    - Заполняет пропуски в текстовых полях.
    - Очищает описание от пунктуации и стоп-слов.
    - Создаёт признаки длины текста, наличия ключевых слов и временные признаки.
    - Преобразует цену в категорию.
    
    Возвращает: модифицированный DataFrame.
    """
    text_columns = ['name', 'description', 'terms', 'section']
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str)
    
    # Очистка текста
    stop_words = set(stopwords.words('english'))
    df['clean_description'] = df['description'].apply(
        lambda x: ' '.join([word for word in re.sub(f'[{string.punctuation}]', '', x.lower()).split() 
                          if word not in stop_words and word != ''])
    )
    
    # Извлечение признаков из текста
    df['desc_length'] = df['clean_description'].apply(len)
    df['word_count'] = df['clean_description'].apply(lambda x: len(x.split()))
    
    # Признаки из названия
    df['name'] = df['name'].fillna('')
    df['name_length'] = df['name'].apply(len)
    df['name_word_count'] = df['name'].apply(lambda x: len(x.split()))
    
    # Признаки цены
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(df['price'].median())
    df['price_category'] = pd.cut(df['price'], bins=[0, 50, 100, 150, 200, np.inf], 
                                 labels=[0, 1, 2, 3, 4], include_lowest=True).astype(int)
    
    # Бинарные признаки
    df['is_jacket'] = df['description'].str.contains('jacket', case=False).fillna(0).astype(int)
    df['has_zip'] = df['description'].str.contains('zip', case=False).fillna(0).astype(int)
    df['has_button'] = df['description'].str.contains('button', case=False).fillna(0).astype(int)
    
    # Временные признаки
    df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
    df['scrape_hour'] = df['scraped_at'].dt.hour.fillna(12)
    df['scrape_day'] = df['scraped_at'].dt.day.fillna(1)
    
    return df


class SalesPredictor(nn.Module):
    """
    Нейросетевая модель для предсказания объема продаж.
    Принимает табличные и текстовые признаки.
    Состоит из трёх блоков: табличного, текстового и объединённого.
    Использует остаточную связь для стабилизации обучения.
    """
    def __init__(self, tabular_size, text_size):
        super().__init__()
        # Блок для табличных данных
        self.tabular_block = nn.Sequential(
            nn.Linear(tabular_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Блок для текста
        self.text_block = nn.Sequential(
            nn.Linear(text_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Комбинированный блок
        self.combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Остаточное соединение
        self.residual = nn.Linear(256, 1)


    def forward(self, x_tab, x_text):
        """
        Выполняет прямой проход данных через модель.
        Возвращает предсказанное значение.
        """
        h_tab = self.tabular_block(x_tab)
        h_text = self.text_block(x_text)
        h_combined = torch.cat([h_tab, h_text], dim=1)
        main_out = self.combined(h_combined)
        res_out = self.residual(h_combined)
        return main_out + res_out


def prepare_features(df, vectorizer=None, preprocessor=None, fit=True):
    """
    Подготавливает табличные и текстовые признаки для модели:
    - Обрабатывает категориальные и числовые признаки с помощью ColumnTransformer.
    - Преобразует описание в TF-IDF-векторы.

    Возвращает: табличные признаки, текстовые признаки, целевую переменную, 
    обученный ColumnTransformer, обученный TfidfVectorizer.
    """
    # Категориальные признаки
    cat_features = ['Product Position', 'Promotion', 'Product Category', 
                   'Seasonal', 'section', 'price_category']
    
    # Числовые признаки
    num_features = ['price', 'desc_length', 'word_count', 'name_length', 
                   'name_word_count', 'is_jacket', 'has_zip', 'has_button',
                   'scrape_hour', 'scrape_day']
    
    if preprocessor is None:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_features),
                ('cat', cat_pipeline, cat_features)
            ]
        )
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    
    if fit:
        X_tab = preprocessor.fit_transform(df)
        X_text = vectorizer.fit_transform(df['clean_description']).toarray()
        return X_tab, X_text, df['Sales Volume'].values, preprocessor, vectorizer
    else:
        X_tab = preprocessor.transform(df)
        X_text = vectorizer.transform(df['clean_description']).toarray()
        return X_tab, X_text, df['Sales Volume'].values


def train_with_cv(df, n_splits=3, epochs=100, batch_size=64):
    """
    Обучает модель с использованием k-fold кросс-валидации.
    Возвращает: список обученных моделей, список препроцессоров, список векторизаторов.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_val_preds = []
    all_val_true = []
    models = []
    preprocessors = []
    vectorizers = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"Fold {fold+1}/{n_splits}")
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        X_train_tab, X_train_text, y_train, preprocessor, vectorizer = prepare_features(train_df, fit=True)
        X_val_tab, X_val_text, y_val = prepare_features(val_df, 
                                                      preprocessor=preprocessor, 
                                                      vectorizer=vectorizer, 
                                                      fit=False)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_tab, dtype=torch.float32),
            torch.tensor(X_train_text, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_tab, dtype=torch.float32),
            torch.tensor(X_val_text, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        model = SalesPredictor(X_train_tab.shape[1], X_train_text.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.HuberLoss()
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_tab, x_text, y in train_loader:
                x_tab, x_text, y = x_tab.to(device), x_text.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x_tab, x_text).squeeze()
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Валидация
            model.eval()
            val_preds = []
            with torch.no_grad():
                for x_tab, x_text, y in val_loader:
                    x_tab, x_text = x_tab.to(device), x_text.to(device)
                    outputs = model(x_tab, x_text).squeeze().cpu().numpy()
                    val_preds.extend(outputs)
            
            val_loss = mean_absolute_error(y_val, val_preds)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val MAE: {val_loss:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
        
        all_val_preds.extend(val_preds)
        all_val_true.extend(y_val)
        models.append(model)
        preprocessors.append(preprocessor)
        vectorizers.append(vectorizer)
    
    mae = mean_absolute_error(all_val_true, all_val_preds)
    mape = mean_absolute_percentage_error(all_val_true, all_val_preds) * 100
    print(f"Cross-Validation MAE: {mae:.2f}")
    print(f"Cross-Validation MAPE: {mape:.2f}%")
    
    return models, preprocessors, vectorizers


def predict_test(test_df, models, preprocessors, vectorizers):
    """
    Делает предсказания на тестовом датасете с использованием ансамбля обученных моделей.
    Возвращает: среднее предсказание по всем моделям.
    """
    all_preds = []
    
    for model, preprocessor, vectorizer in zip(models, preprocessors, vectorizers):
        X_test_tab, X_test_text, _ = prepare_features(
            test_df, 
            preprocessor=preprocessor,
            vectorizer=vectorizer,
            fit=False
        )
        
        test_dataset = TensorDataset(
            torch.tensor(X_test_tab, dtype=torch.float32),
            torch.tensor(X_test_text, dtype=torch.float32)
        )
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for x_tab, x_text in test_loader:
                x_tab, x_text = x_tab.to(device), x_text.to(device)
                outputs = model(x_tab, x_text).squeeze().cpu().numpy()
                fold_preds.append(outputs)
        
        all_preds.append(np.concatenate(fold_preds))
    
    avg_preds = np.mean(all_preds, axis=0)
    
    results = pd.DataFrame({
        'Actual': test_df['Sales Volume'].values,
        'Predicted': avg_preds
    })
    
    return results


def plot_results(results):
    """
    Визуализирует предсказания модели:
    - Предсказания vs Истинные значения.
    - Распределение ошибок (гистограмма).
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(results['Actual'], results['Predicted'], alpha=0.6)
    max_val = max(results['Actual'].max(), results['Predicted'].max()) + 100
    min_val = min(results['Actual'].min(), results['Predicted'].min()) - 100
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Sales Volume')
    plt.ylabel('Predicted Sales Volume')
    plt.title('Actual vs Predicted Sales Volume')
    plt.grid(True, alpha=0.3)
    plt.savefig('actual_vs_predicted.png')
    plt.show()
    
    results['Error'] = results['Predicted'] - results['Actual']
    plt.figure(figsize=(10, 6))
    plt.scatter(results['Actual'], results['Error'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Actual Sales Volume')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error by Sales Volume')
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_error.png')
    plt.show()


def main():
    """
    Главная функция пайплайна:
    - Загружает и подготавливает данные.
    - Обучает модель с кросс-валидацией.
    - Делает предсказания на тесте.
    - Визуализирует результаты.
    """
    df = pd.read_csv('C:/Users/Lerik/OneDrive/Desktop/all_practices/zara.csv', sep=';')
    df['Sales Volume'] = df['Sales Volume'].fillna(0)
    
    df = preprocess_data(df)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print("Starting cross-validation training...")
    models, preprocessors, vectorizers = train_with_cv(train_df, n_splits=3, epochs=50)
    
    print("Predicting on test set...")
    test_results = predict_test(test_df, models, preprocessors, vectorizers)
    print(test_results.head(20))
    
    mae = mean_absolute_error(test_results['Actual'], test_results['Predicted'])
    mape = mean_absolute_percentage_error(test_results['Actual'], test_results['Predicted']) * 100
    print(f"\nTest MAE: {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")
    rmse = rmse = np.sqrt(mean_squared_error(test_results['Actual'], test_results['Predicted']))
    r2 = r2_score(test_results['Actual'], test_results['Predicted'])
    smape = np.mean(2 * np.abs(test_results['Predicted'] - test_results['Actual']) /
                (np.abs(test_results['Predicted']) + np.abs(test_results['Actual']))) * 100

    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test SMAPE: {smape:.2f}%")

    plot_results(test_results)

if __name__ == "__main__":
    main()
