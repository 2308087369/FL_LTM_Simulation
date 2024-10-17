import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 准备数据集 data_AT
def load_data_AT():
    # 假设 specific_length_dfs['Preprocess_data/data_AT'] 是已经处理好的数据集
    data_at = pd.read_csv('Preprocess_data/data_AT.csv')
    X, y = [], []
    window_size = 10
    for i in range(len(data_at) - window_size):
        X.append(data_at.iloc[i:i+window_size][['Year', 'Month', 'Day', 'Hour', 'Weekday', 'Value']].values)
        y.append(data_at.iloc[i+window_size]['Value'])
    X = np.array(X)
    y = np.array(y)
    
    split_point = int(len(X) * 0.7)
    return X[:split_point], X[split_point:], y[:split_point], y[split_point:]

# 创建LSTM模型
def create_model_AT():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(10, 6), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Flower 客户端
class ATClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_loss_history = []  # 保存训练损失

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.y_train, epochs=1, verbose=0)
        train_loss = history.history['loss'][0]
        self.train_loss_history.append(train_loss)
        print(f"训练损失: {train_loss}")
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"评估损失: {loss}")
        return loss, len(self.X_test), {"loss": loss}

    def plot_training_loss(self):
        """绘制训练损失曲线并保存为SVG格式"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label="Train Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("train_loss_curve.svg", format="svg")  # 保存为SVG格式
        print("训练损失曲线已保存为 train_loss_curve.svg")

    def plot_predictions(self):
        """绘制实际值与预测值的对比图以及相对误差图，保存为SVG格式"""
        predictions = self.model.predict(self.X_test)
        plt.figure(figsize=(15, 5))
        
        # 实际值和预测值的对比图
        plt.subplot(1, 2, 1)
        plt.plot(self.y_test, label="Actual Values")
        plt.plot(predictions, label="Predicted Values", linestyle='--')
        plt.xlabel("Test Samples")
        plt.ylabel("Value")
        plt.title("Actual vs Predicted")
        plt.legend()

        # 相对误差图
        relative_error = np.abs((self.y_test - predictions.flatten()) / self.y_test)
        plt.subplot(1, 2, 2)
        plt.plot(relative_error, label="Relative Error")
        plt.xlabel("Test Samples")
        plt.ylabel("Relative Error")
        plt.title("Relative Error (Absolute)")
        plt.legend()

        plt.savefig("predictions_vs_actual_and_error.svg", format="svg")  # 保存为SVG格式
        print("预测效果图和相对误差图已保存为 predictions_vs_actual_and_error.svg")

if __name__ == "__main__":
    # 使用 data_AT
    X_train, X_test, y_train, y_test = load_data_AT()
    
    # 不指定设备，默认使用 CPU
    model_AT = create_model_AT()
    client_AT = ATClient(model_AT, X_train, y_train, X_test, y_test)

    # 启动客户端
    fl.client.start_numpy_client(server_address="localhost:8080", client=client_AT)

    # 在训练完成后绘制并保存训练损失曲线
    client_AT.plot_training_loss()

    # 在训练完成后绘制并保存预测效果图和相对误差图
    client_AT.plot_predictions()
