import os
import subprocess
import platform

def open_svg(file_name):
    """根据操作系统自动打开 SVG 文件"""
    try:
        if platform.system() == 'Windows':  # Windows
            os.startfile(file_name)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', file_name])
        elif platform.system() == 'Linux':  # Linux
            subprocess.run(['xdg-open', file_name])
        else:
            print("不支持的操作系统")
    except Exception as e:
        print(f"无法打开文件: {file_name}, 错误: {e}")

if __name__ == "__main__":
    # 需要查看的 SVG 文件
    svg_files = ["train_loss_curve.svg", "predictions_vs_actual_and_error.svg"]

    for svg_file in svg_files:
        print(f"正在打开 {svg_file} ...")
        open_svg(svg_file)
