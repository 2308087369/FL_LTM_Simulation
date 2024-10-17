import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import psutil

# 检查端口是否被占用
def check_and_clear_port(port):
    """检查并终止占用指定端口的进程"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            pid = conn.pid
            if pid:
                try:
                    process = psutil.Process(pid)
                    print(f"端口 {port} 已被进程 {process.name()} (PID: {pid}) 占用，正在终止进程...")
                    process.terminate()  # 终止占用端口的进程
                    process.wait()  # 等待进程终止
                    print(f"进程 {pid} 已成功终止。")
                except Exception as e:
                    print(f"无法终止进程 PID {pid}：{e}")
            else:
                print(f"端口 {port} 被占用，但无法获取进程信息。")

# 创建一个初始 LSTM 模型，用于初始化服务器端的全局模型
def create_initial_model():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(10, 6), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 获取模型的初始权重
def get_initial_parameters():
    model = create_initial_model()
    return model.get_weights()

# 自定义策略，初始化全局模型的初始权重
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, **kwargs):
        super().__init__(**kwargs)
        self.initial_parameters = initial_parameters

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

if __name__ == "__main__":
    # 创建初始模型并获取其初始参数
    initial_parameters = get_initial_parameters()
        # 设定默认端口
    port = 8080

    # 检查并清除端口占用
    check_and_clear_port(port)
    # 创建自定义的策略
    strategy = CustomFedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),  # 将初始模型的权重转为 Flower 参数格式
        min_fit_clients=2,  # 每轮最小的训练客户端数
        min_available_clients=2  # 每轮最小可用客户端数
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=10),  # 这里设置训练轮数
        strategy=strategy  # 使用自定义策略
    )
