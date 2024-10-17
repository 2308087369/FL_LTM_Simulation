This is a simulation project:
# FL_LTM_Simulation

## 项目简介

本项目旨在使用联邦学习 (Federated Learning, FL) 的思想，结合长短时记忆网络 (LSTM) 模型，对电力负载进行预测。通过模拟多个不同节点的数据共享与训练，实现高效、隐私保护的电力负载预测。该项目主要应用于分布式系统的负载预测，特别是不同区域电力负载的数据。

## 项目结构

FL_LTM_Simulation/ │ ├── client_AT.py # 代表节点 AT 的客户端代码 ├── client_BE.py # 代表节点 BE 的客户端代码 ├── start_server.py # 启动 Flower 服务器的代码 ├── Preprocessed_data/ # 存放预处理后的数据文件夹 │ ├── data_AT.csv # AT 国家预处理后的数据 │ └── data_BE.csv # BE 国家预处理后的数据 ├── view_svg.py # 快速打开生成的 SVG 图像文件 ├── README.md # 项目说明文档 └── requirements.txt # 项目依赖的 Python 库

## 数据集

本项目使用了处理后的电力负载数据集，数据已存储在 `Preprocessed_data/` 文件夹下：

- `data_AT.csv`: AT 国家电力负载数据
- `data_BE.csv`: BE 国家电力负载数据

### 数据字段：

- `Year`：年份
- `Month`：月份
- `Day`：日期
- `Hour`：小时
- `Weekday`：星期几
- `Value`：每小时的负载值
