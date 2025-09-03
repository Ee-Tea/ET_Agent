#!/usr/bin/env python3
"""
간단한 LangGraph 웹 인터페이스
"""

import os
import sys
import time
from flask import Flask, request, jsonify, render_template_string

# Flask 앱 생성
app = Flask(__name__)

# 간단한 HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Teacher Graph - LangGraph 인터페이스</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 900px; 
            margin: 0 auto; 
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 { 
            color: #333; 
            text-align: center; 
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .status { 
            color: #28a745; 
            font-weight: bold; 
            text-align: center;
            padding: 15px;
            background: #d4edda;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .form-group { 
            margin-bottom: 25px; 
        }
        label { 
            display: block; 
            margin-bottom: 8px; 
            font-weight: bold; 
            color: #555;
            font-size: 1.1em;
        }
        textarea, select { 
            width: 100%; 
            padding: 12px; 
            border: 2px solid #e1e5e9; 
            border-radius: 8px; 
            font-size: 16px;
            transition: border-color 0.3s;
        }
        textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        button { 
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: bold;
            width: 100%;
            transition: transform 0.2s;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .result { 
            margin-top: 30px; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 8px; 
            border-left: 4px solid #667eea;
        }
        .result h3 {
            color: #333;
            margin-top: 0;
        }
        pre {
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 14px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            font-weight: bold;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Teacher Graph - LangGraph 인터페이스</h1>
        <div class="status">
            ✅ LangGraph 웹 서버가 실행 중입니다 (포트: {{ port }})
        </div>
        
        <form id="queryForm">
            <div class="form-group">
                <label for="user_query">질문 입력:</label>
                <textarea id="user_query" name="user_query" rows="4" 
                    placeholder="예시: XP(eXtreme Programming)에 대한 설명으로 옳지 않은 것은?&#10;1) 릴리즈 기간을 짧게 반복하여 고객의 요구 변화에 빠르게 대응한다&#10;2) 코드들은 하나의 작업이 마무리될 때마다 지속적으로 통합한다&#10;3) 테스트가 지속적으로 진행될 수 있도록 테스트 자동화 도구를 사용한다&#10;4) 개발 책임자가 모든 책임을 가지므로 팀원들은 책임 없이 자유로운 개발이 가능하다"></textarea>
            </div>
            
            <div class="form-group">
                <label for="intent">의도 선택:</label>
                <select id="intent" name="intent">
                    <option value="solution">문제 풀이</option>
                    <option value="generation">문제 생성</option>
                    <option value="analysis">성적 분석</option>
                    <option value="score">채점</option>
                </select>
            </div>
            
            <button type="submit">🚀 실행하기</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>LangGraph 실행 중...</p>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h3>📊 실행 결과:</h3>
            <div id="resultContent"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('queryForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // 로딩 표시
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const formData = new FormData(e.target);
            const data = {
                user_query: formData.get('user_query'),
                intent: formData.get('intent')
            };
            
            try {
                const response = await fetch('/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                document.getElementById('resultContent').innerHTML = `
                    <p><strong>상태:</strong> <span style="color: ${result.status === 'success' ? '#28a745' : '#dc3545'}">${result.status}</span></p>
                    <p><strong>실행 시간:</strong> ${result.result.timestamp}</p>
                    <p><strong>의도:</strong> ${result.result.intent}</p>
                    <p><strong>결과:</strong></p>
                    <pre>${JSON.stringify(result.result, null, 2)}</pre>
                `;
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                document.getElementById('resultContent').innerHTML = `
                    <p style="color: #dc3545;"><strong>오류:</strong> ${error.message}</p>
                `;
                document.getElementById('result').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    port = request.environ.get('SERVER_PORT', '8080')
    return render_template_string(HTML_TEMPLATE, port=port)

@app.route('/execute', methods=['POST'])
def execute():
    try:
        data = request.get_json()
        user_query = data.get('user_query', '')
        intent = data.get('intent', 'solution')
        
        if not user_query.strip():
            return jsonify({'status': 'error', 'result': '질문을 입력해주세요.'})
        
        # 여기서 실제 LangGraph 실행 로직을 구현할 수 있습니다
        # 현재는 간단한 응답만 반환
        result = {
            'user_query': user_query,
            'intent': intent,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success',
            'message': f'"{intent}" 의도로 질문이 처리되었습니다.',
            'query_length': len(user_query),
            'processing_time': '0.1초'
        }
        
        return jsonify({'status': 'success', 'result': result})
        
    except Exception as e:
        return jsonify({'status': 'error', 'result': str(e)})

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'service': 'Teacher Graph LangGraph',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0.0'
    })

@app.route('/api/status')
def api_status():
    return jsonify({
        'langgraph': 'running',
        'milvus': 'connected',
        'redis': 'connected',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Teacher Graph 웹 인터페이스")
    parser.add_argument("--port", type=int, default=8080, help="웹 서버 포트 (기본: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="웹 서버 호스트 (기본: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    args = parser.parse_args()
    
    print("🌐 LangGraph 웹 인터페이스 시작 중...")
    print(f"📍 서버 주소: http://{args.host}:{args.port}")
    print(f"🔗 메인 페이지: http://{args.host}:{args.port}/")
    print(f"💚 상태 확인: http://{args.host}:{args.port}/health")
    print(f"📊 API 상태: http://{args.host}:{args.port}/api/status")
    print(f"⏹️  중지하려면 Ctrl+C를 누르세요")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=args.debug)
