# -*- coding: utf-8 -*-
"""局域网问卷服务器：手机/电脑浏览器访问即填，提交自动保存。

用法: python survey_server.py
      (确保被试与本机在同一 WiFi/局域网)
      访问 http://<本机IP>:5000  （启动时会打印）
提交的结果自动保存到 results_submitted/ 目录。

依赖: pip install flask
前置: 先运行 make_survey.py 生成 survey.html
"""
import sys
import os
import socket
import datetime
import json

sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, 'results_submitted')
SURVEY = os.path.join(BASE, 'survey.html')
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)


def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


@app.route('/')
def index():
    """直接提供 make_survey.py 生成的问卷（其 JS 会 POST 到 /submit）。"""
    if not os.path.exists(SURVEY):
        return ('survey.html 不存在，请先运行: python make_survey.py', 500)
    with open(SURVEY, encoding='utf-8') as f:
        return f.read()


@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json(force=True)
    name = (data.get('name') or 'anonymous').strip()
    pairs = data.get('pairs') or []

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    obs_id = f'{name}_{ts}'
    fn = os.path.join(SAVE_DIR, f'{obs_id}.txt')

    with open(fn, 'w', encoding='utf-8') as f:
        f.write(f'observer: {name}\n')
        f.write(f'提交时间: {ts}\n')
        f.write(f'试次数: {len(pairs)}\n')
        f.write('编号,选择\n')
        for p in pairs:
            f.write(str(p) + '\n')

    print(f'[保存] {fn}  ({len(pairs)} 试次)')
    return jsonify({'obs_id': obs_id, 'ok': True})


if __name__ == '__main__':
    ip = get_lan_ip()
    print('=' * 52)
    print('主观实验问卷服务器已启动')
    print(f'  被试访问: http://{ip}:5000')
    print(f'  本机测试: http://127.0.0.1:5000')
    print(f'  结果保存: {SAVE_DIR}')
    print('  按 Ctrl+C 停止')
    print('=' * 52)
    app.run(host='0.0.0.0', port=5000, debug=False)
