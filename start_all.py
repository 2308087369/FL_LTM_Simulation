import subprocess
import time

# 启动服务器
server_process = subprocess.Popen(["python", "start_server.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)

# 等待 10 秒钟，确保服务器完全启动
time.sleep(10)

# 启动第一个客户端
client_at_process = subprocess.Popen(["python", "client_AT.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
# 启动第二个客户端
client_be_process = subprocess.Popen(["python", "client_BE.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)

# 可选：等待所有进程完成
server_process.wait()
client_at_process.wait()
client_be_process.wait()
