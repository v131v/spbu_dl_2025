"""
Year Prediction MSD - Best Solution
Максимальное качество с длительным обучением

Улучшения:
- Очень большая модель: [1024, 512, 384, 256, 128, 64]
- Много эпох: 500
- Большой patience: 50
- Меньший learning rate для более точной настройки
- Дополнительные residual connections
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

device = torch.device('cpu')
print('='*60)
print('Year Prediction MSD - Best Solution')
print('='*60)
print(f'Device: {device}')
print('Maximum quality with extended training')
print('Expected training time: 5-10 minutes on CPU')
print('='*60)


class BestYearPredictionNet(nn.Module):
    """
    Максимально мощная нейронная сеть
    Архитектура: 90 → 1024 → 512 → 384 → 256 → 128 → 64 → 1
    """
    def __init__(self, input_dim=90, hidden_dims=[1024, 512, 384, 256, 128, 64], dropout_rate=0.35):
        super(BestYearPredictionNet, self).__init__()

        # Входной блок
        self.input_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Скрытые блоки
        self.hidden_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.hidden_blocks.append(block)

        # Множественные residual projections
        self.residual_proj1 = nn.Linear(hidden_dims[0], hidden_dims[2])  # 1024 -> 384
        self.residual_proj2 = nn.Linear(hidden_dims[2], hidden_dims[4])  # 384 -> 128
        self.residual_proj3 = nn.Linear(hidden_dims[4], hidden_dims[5])  # 128 -> 64

        # Выходной слой
        self.output = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # Входной блок
        out = self.input_block(x)
        identity1 = out

        # Первые два блока
        out = self.hidden_blocks[0](out)  # 1024 -> 512
        out = self.hidden_blocks[1](out)  # 512 -> 384

        # Первое residual connection
        identity1_proj = self.residual_proj1(identity1)
        out = out + identity1_proj
        identity2 = out

        # Следующие блоки
        out = self.hidden_blocks[2](out)  # 384 -> 256
        out = self.hidden_blocks[3](out)  # 256 -> 128

        # Второе residual connection
        identity2_proj = self.residual_proj2(identity2)
        out = out + identity2_proj
        identity3 = out

        # Последний блок
        out = self.hidden_blocks[4](out)  # 128 -> 64

        # Третье residual connection
        identity3_proj = self.residual_proj3(identity3)
        out = out + identity3_proj

        # Выход
        out = self.output(out)
        return out.squeeze()


class EarlyStopping:
    """Early stopping с большим patience"""
    def __init__(self, patience=50, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def load_and_preprocess_data(data_dir):
    """Загрузка и предобработка данных"""
    print("\nLoading data...")

    train_x = pd.read_csv(os.path.join(data_dir, 'train_x.csv'), index_col=0)
    train_y = pd.read_csv(os.path.join(data_dir, 'train_y.csv'), index_col=0)
    test_x_df = pd.read_csv(os.path.join(data_dir, 'test_x.csv'))

    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")
    print(f"Test X shape: {test_x_df.shape}")

    X_train_full = train_x.values
    y_train_full = train_y['year'].values

    # Извлекаем ID и признаки из test_x
    test_ids = test_x_df['id'].values
    X_test = test_x_df.drop('id', axis=1).values

    print(f"\nYear range: {y_train_full.min():.0f} - {y_train_full.max():.0f}")
    print(f"Mean year: {y_train_full.mean():.2f}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42
    )

    # Стандартизация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Test IDs range: {test_ids.min()} - {test_ids.max()}")

    return X_train, X_val, X_test, y_train, y_val, test_ids, scaler


def create_dataloaders(X_train, X_val, y_train, y_val, batch_size=128):
    """Создание DataLoaders"""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, show_progress=True):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0

    if show_progress:
        pbar = tqdm(train_loader, desc='Training', leave=False)
    else:
        pbar = train_loader

    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        if show_progress and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    """Валидация модели"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(val_loader.dataset)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs, device, model_path):
    """Полный цикл обучения"""
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=50, min_delta=0.0005)

    train_losses = []
    val_losses = []

    print("\nStarting training...")
    print(f"Total epochs: {num_epochs}")
    print(f"Early stopping patience: 50")
    print()

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Показываем прогресс только каждые 10 эпох для ускорения
        show_progress = (epoch % 10 == 0)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, show_progress)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time

        # Выводим каждые 5 эпох
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] '
                  f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
                  f'LR: {current_lr:.6f} | Time: {epoch_time:.1f}s | '
                  f'Total: {elapsed_time/60:.1f}m')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            if (epoch + 1) % 5 == 0 or epoch < 20:
                print(f'  ✓ Model saved! Best Val Loss: {val_loss:.4f} (RMSE: {np.sqrt(val_loss):.4f})')

        if early_stopping(val_loss):
            print(f'\n⚠️  Early stopping at epoch {epoch+1}')
            break

    total_time = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'Training completed in {total_time/60:.1f} minutes')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation RMSE: {np.sqrt(best_val_loss):.4f} years')
    print(f'{"="*60}')

    return train_losses, val_losses


def predict(model, X_test, device, batch_size=128):
    """Генерация предсказаний"""
    model.eval()
    predictions = []

    X_test_tensor = torch.FloatTensor(X_test)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("\nGenerating predictions...")
    with torch.no_grad():
        for (X_batch,) in tqdm(test_loader, desc='Predicting'):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)


def create_submission(predictions, test_ids, output_path='submission.csv'):
    """Создание submission файла с правильным форматом для Kaggle

    Формат: index,year
    - index: ID из test_x.csv
    - year: целое число (округленный год)
    """
    # Округляем предсказания до целых чисел
    predictions_int = np.round(predictions).astype(int)

    submission = pd.DataFrame({
        'id': test_ids,  # Правильное название колонки для Kaggle
        'year': predictions_int  # Целые числа
    })
    submission.to_csv(output_path, index=False)
    print(f'\n✓ Submission saved: {output_path}')
    print(f'  Total predictions: {len(predictions)}')
    print(f'  Year range: {predictions_int.min()} - {predictions_int.max()}')
    print(f'  Index range: {test_ids.min()} - {test_ids.max()}')


def main():
    """Основная функция"""

    data_dir = 'data'

    required_files = ['train_x.csv', 'train_y.csv', 'test_x.csv']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]

    if missing_files:
        print(f"\n❌ Error: Missing files: {', '.join(missing_files)}")
        return

    # Загрузка данных
    X_train, X_val, X_test, y_train, y_val, test_ids, scaler = load_and_preprocess_data(data_dir)

    # DataLoaders
    batch_size = 128
    train_loader, val_loader = create_dataloaders(
        X_train, X_val, y_train, y_val, batch_size=batch_size
    )

    # Максимальная модель
    model = BestYearPredictionNet(
        input_dim=90,
        hidden_dims=[1024, 512, 384, 256, 128, 64],
        dropout_rate=0.35
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BestYearPredictionNet")
    print(f"Total parameters: {total_params:,}")
    print(f"Architecture: 90 → 1024 → 512 → 384 → 256 → 128 → 64 → 1")

    # Optimizer и scheduler с меньшим LR
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    # Обучение (много эпох)
    model_path = 'models/best_model_final.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=500,  # Много эпох
        device=device,
        model_path=model_path
    )

    # Загрузка лучшей модели
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']+1}")

    # Предсказания
    predictions = predict(model, X_test, device, batch_size=batch_size)

    # Submission
    submission_path = 'submission_best.csv'
    create_submission(predictions, test_ids, submission_path)

    print("\n" + "="*60)
    print("✓ Training completed successfully!")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Submission: {submission_path}")
    print("\n🎉 Ready to submit to Kaggle!")
    print("="*60)


if __name__ == '__main__':
    main()
