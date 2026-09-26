# -*- coding: utf-8 -*-
"""从 trials.csv + pairs/ 生成主观实验问卷页面（survey.html）。

特性：
  - 图片以 base64 内嵌，单文件可分发给被试（无需服务器）
  - 强制 2AFC：仅 A / B 两个选项，无"差不多"
  - 被试 ID 输入 + 44 试次 + 进度显示
  - 本地保存为 CSV 下载；配合 survey_server.py 可自动上传
  - 键盘快捷键：A / B 选择，← → 翻页

用法: python make_survey.py
输出: survey.html
"""
import os
import sys
import csv
import base64

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(BASE, 'pairs')
TRIALS = os.path.join(BASE, 'trials.csv')
OUT = os.path.join(BASE, 'survey.html')

INSTRUCTION = """您将看到成对的图片。每对中，左图标记为 <b>A</b>，右图标记为 <b>B</b>。
两张图是同一场景经不同方式压缩颜色后的结果。<br><br>
请判断<b>哪一张看起来更好</b>（更接近真实场景、色彩更自然、瑕疵更少）。<br>
<span class="hint">必须二选一，没有"差不多"选项。若难以判断，请凭第一印象选择。</span>"""


def load_trials():
    with open(TRIALS, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def main():
    trials = load_trials()
    n = len(trials)

    blocks = []
    for t in trials:
        tid = t['trial_id']
        src = os.path.join(PAIRS, t['file'])
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        img = b64(src)
        blocks.append(f'''
<div class="pair" id="pair-{tid}" data-tid="{tid}">
  <h3>第 {tid} / {n} 组</h3>
  <img src="data:image/png;base64,{img}" alt="trial {tid}">
  <div class="choices">
    <label><input type="radio" name="q{tid}" value="A"> A 更好</label>
    <label><input type="radio" name="q{tid}" value="B"> B 更好</label>
  </div>
</div>''')

    nids = ','.join(t['trial_id'] for t in trials)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图像质量主观评测（2AFC）</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         max-width: 980px; margin: 0 auto; padding: 20px 16px 80px;
         background: #fafafa; color: #222; }}
  h1 {{ font-size: 22px; }}
  .intro {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
            padding: 16px 18px; margin-bottom: 22px; line-height: 1.7; }}
  .hint {{ color: #888; font-size: 13px; }}
  .pair {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
           padding: 14px; margin-bottom: 20px; }}
  .pair h3 {{ margin: 0 0 10px; color: #555; font-size: 14px; font-weight: 600; }}
  .pair img {{ width: 100%; display: block; border: 1px solid #ddd;
               border-radius: 6px; }}
  .choices {{ display: flex; gap: 28px; margin-top: 12px; font-size: 15px; }}
  .choices label {{ cursor: pointer; user-select: none; }}
  .choices input {{ margin-right: 6px; transform: scale(1.15); }}
  .pair.done {{ border-color: #1a73e8; }}
  #bar {{ position: fixed; left: 0; bottom: 0; width: 100%; height: 42px;
          background: #1a73e8; color: #fff; display: flex; align-items: center;
          justify-content: center; font-size: 14px; gap: 18px; z-index: 10; }}
  #bar button {{ padding: 8px 22px; font-size: 15px; border: none;
                 border-radius: 6px; background: #fff; color: #1a73e8;
                 cursor: pointer; font-weight: 600; }}
  #bar button:disabled {{ opacity: .45; cursor: default; }}
  input[type=text] {{ width: 46%; padding: 9px; font-size: 15px;
                      border: 1px solid #ccc; border-radius: 6px; }}
  #result {{ display: none; background: #e8f5e9; border: 1px solid #a5d6a7;
             border-radius: 10px; padding: 18px; margin-bottom: 20px;
             line-height: 1.7; }}
</style>
</head>
<body>
<h1>图像质量主观评测</h1>
<div class="intro">
  <p>{INSTRUCTION}</p>
  <p>共 <b>{n}</b> 组，约需 20 分钟。中途可关闭页面稍后重填。</p>
  <p><b>您的编号：</b><input type="text" id="observer"
     placeholder="例如：S01" required></p>
  <p class="hint">键盘操作：按 <b>A</b> / <b>B</b> 快速选择，<b>↑</b> / <b>↓</b> 跳转。</p>
</div>
<div id="result"></div>

{''.join(blocks)}

<div id="bar">
  <span id="progress">已完成 0 / {n}</span>
  <button id="submitBtn" onclick="submitForm()">提交</button>
</div>

<script>
var NIDS = [{nids}];
var TOTAL = NIDS.length;

function answered(tid) {{
  var sel = document.querySelector('input[name="q' + tid + '"]:checked');
  return sel ? sel.value : null;
}}

function refresh() {{
  var done = 0;
  for (var i = 0; i < TOTAL; i++) {{
    var tid = NIDS[i];
    var v = answered(tid);
    var el = document.getElementById('pair-' + tid);
    if (v) {{ done++; el.classList.add('done'); }}
    else {{ el.classList.remove('done'); }}
  }}
  document.getElementById('progress').textContent =
    '已完成 ' + done + ' / ' + TOTAL;
}}

document.addEventListener('change', function(e) {{
  if (e.target.name && e.target.name.charAt(0) === 'q') refresh();
}});

document.addEventListener('keydown', function(e) {{
  if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
  var k = e.key.toUpperCase();
  if (k !== 'A' && k !== 'B') return;
  // 找到当前视口内最近的一组
  var best = null, bestD = 1e9;
  for (var i = 0; i < TOTAL; i++) {{
    var el = document.getElementById('pair-' + NIDS[i]);
    var r = el.getBoundingClientRect();
    var d = Math.abs(r.top);
    if (d < bestD) {{ bestD = d; best = NIDS[i]; }}
  }}
  if (best === null) return;
  var input = document.querySelector(
    'input[name="q' + best + '"][value="' + k + '"]');
  if (input) {{ input.checked = true; refresh(); }}
}});

function collect() {{
  var pairs = [];
  for (var i = 0; i < TOTAL; i++) {{
    var tid = NIDS[i];
    var v = answered(tid);
    if (!v) return null;
    pairs.push(tid + ',' + v);
  }}
  return pairs;
}}

function submitForm() {{
  var name = (document.getElementById('observer').value || '').trim();
  if (!name) {{ alert('请先填写您的编号。'); return; }}
  var pairs = collect();
  if (!pairs) {{ alert('还有未作答的组，请检查后提交。'); return; }}

  // 若由 survey_server.py 提供，则上传；否则本地下载 CSV
  fetch('/submit', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{name: name, pairs: pairs}})
  }}).then(function(r) {{ return r.json(); }}).then(function(d) {{
    showResult(name, d.obs_id);
  }}).catch(function() {{
    downloadCSV(name, pairs);
  }});
}}

function downloadCSV(name, pairs) {{
  var lines = ['observer,trial_id,choice'];
  for (var i = 0; i < pairs.length; i++) {{
    var p = pairs[i].split(',');
    lines.push(name + ',' + p[0] + ',' + p[1]);
  }}
  var blob = new Blob([lines.join('\\n')], {{type: 'text/csv'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'answers_' + name + '.csv';
  a.click();
  showResult(name, '本地CSV已下载');
}}

function showResult(name, ref) {{
  var div = document.getElementById('result');
  div.style.display = 'block';
  div.innerHTML = '<b>✅ 提交成功，感谢参与！</b><br>编号：' + name +
                  '<br>记录：' + ref +
                  '<br><br>请将结果交给实验主试，然后关闭本页。';
  window.scrollTo({{top: 0, behavior: 'smooth'}});
}}

refresh();
</script>
</body>
</html>
'''

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(OUT) / 1024 / 1024
    print(f'生成 {OUT}')
    print(f'  {n} 试次，单文件 {size_mb:.1f} MB（图片已内嵌，可直接分发）')


if __name__ == '__main__':
    main()
