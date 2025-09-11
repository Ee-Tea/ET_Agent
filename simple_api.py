"""
ET-Agent FastAPI 서버
실제 ET-Agent 백엔드와 연결된 버전
"""

import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import glob

from auth.auth_routes import router as auth_router

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ET-Agent 메인 오케스트레이터 import
try:
    from supervisor import MainOrchestrator    
    ET_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ET-Agent import failed: {e}")
    ET_AGENT_AVAILABLE = False

# 전역 변수
orchestrator = None
pdf_generation_status = {
    "is_generating": False,
    "last_generated_time": None,
    "generated_files": []
}

# FastAPI 앱 생성
app = FastAPI(
    title="ET-Agent API",
    description="농업 자격증 및 교육 관련 AI 에이전트 API",
    version="1.0.0"
)

# CORS 미들웨어 설정
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(auth_router)

# ET-Agent 초기화 함수
def initialize_et_agent():
    """ET-Agent 오케스트레이터 초기화"""
    global orchestrator
    
    if not ET_AGENT_AVAILABLE:
        print("❌ ET-Agent를 사용할 수 없습니다.")
        return False
    
    try:
        print("🚀 ET-Agent 초기화 중...")
        orchestrator = MainOrchestrator(
            user_id="api_user",
            chat_id="api_chat"
        )
        print("✅ ET-Agent 초기화 완료")
        return True
    except Exception as e:
        print(f"❌ ET-Agent 초기화 실패: {e}")
        return False

# 앱 시작 시 ET-Agent 초기화
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    initialize_et_agent()

# Pydantic 모델
class ChatRequest(BaseModel):
    message: str
    user_id: str = "api_user"
    chat_id: str = "api_chat"

class ChatResponse(BaseModel):
    response: str
    service_used: str
    confidence: float
    session_id: str
    grading_results: Optional[Dict[str, Any]] = None

# API 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "ET-Agent Simple API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "et_agent": "available" if ET_AGENT_AVAILABLE and orchestrator else "unavailable",
            "orchestrator": "initialized" if orchestrator else "not_initialized"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """ET-Agent 채팅 엔드포인트"""
    
    # ET-Agent가 사용 가능한 경우
    if orchestrator and ET_AGENT_AVAILABLE:
        # PDF 생성 상태 초기화
        pdf_generation_status["is_generating"] = True
        pdf_generation_status["generated_files"] = []
        
        try:
            # ET-Agent 상태 초기화 (MainState 구조에 맞게)
            initial_state = {
                "user_query": request.message,
                "user_id": request.user_id,
                "chat_id": request.chat_id,
                "session_key": f"{request.user_id}_{request.chat_id}",
                "existing_questions": [],
                "locked_service": None,
                "short_term_data": {},
                "is_relevant": True,
                "classified_service": "",
                "service_consistent": True,
                "teacher_result": None,
                "farmer_result": None,
                "final_response": ""
            }
            
            # ET-Agent 실행 (체크포인터 설정 포함)
            config = {
                "configurable": {
                    "thread_id": f"api_{request.user_id}_{request.chat_id}",
                    "checkpoint_id": "api_checkpoint"
                }
            }
            result = orchestrator.graph.invoke(initial_state, config=config)
            
            # PDF 생성 상태 업데이트
            pdf_generation_status["is_generating"] = False
            pdf_generation_status["last_generated_time"] = time.time()
            
            # 새로 생성된 PDF 파일 확인
            import glob
            current_time = time.time()
            pdf_dirs = [
                "teacher/agents/solution/pdf_outputs",
                "teacher/pdf_outputs", 
                "farmer/pdf_outputs"
            ]
            
            new_pdfs = []
            for pdf_dir in pdf_dirs:
                if os.path.exists(pdf_dir):
                    files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
                    for file_path in files:
                        file_created_time = os.path.getctime(file_path)
                        # 최근 5분 내에 생성된 파일만 포함
                        if file_created_time >= current_time - 300:
                            new_pdfs.append(os.path.basename(file_path))
            
            pdf_generation_status["generated_files"] = new_pdfs
            
            # 응답 구성
            response_text = result.get("final_response", "응답을 생성할 수 없습니다.")
            
            # 채점 결과 추출 (채점 요청인 경우)
            grading_results = None
            print(f"🔍 [API] 채점 결과 추출 시작")
            print(f"🔍 [API] result 키들: {list(result.keys())}")
            print(f"🔍 [API] score 존재 여부: {'score' in result}")
            
            if "score" in result:
                score_data = result["score"]
                print(f"🔍 [API] score_data: {score_data}")
                print(f"🔍 [API] score status: {score_data.get('status')}")
                
                if score_data.get("status") == "success":
                    shared_data = result.get("shared", {})
                    print(f"🔍 [API] shared_data 키들: {list(shared_data.keys())}")
                    print(f"🔍 [API] user_answer: {shared_data.get('user_answer')}")
                    print(f"🔍 [API] score results: {score_data.get('results')}")
                    
                    # 채점 결과가 있는 경우
                    if "results" in score_data and shared_data.get("user_answer"):
                        user_answers = shared_data["user_answer"]
                        score_results = score_data["results"]
                        
                        print(f"🔍 [API] user_answers: {user_answers}")
                        print(f"🔍 [API] score_results: {score_results}")
                        
                        # 각 문제별 채점 결과 구성
                        grading_results = {}
                        for i, (user_answer, score_result) in enumerate(zip(user_answers, score_results)):
                            if user_answer:  # 답변이 있는 경우만
                                question_id = f"question-{i+1}"
                                is_correct = score_result == 1
                                grading_results[question_id] = {
                                    "isCorrect": is_correct,
                                    "userAnswer": user_answer,
                                    "score": score_result
                                }
                                print(f"🔍 [API] 문제 {i+1}: {user_answer} -> {score_result} ({'정답' if is_correct else '오답'})")
                        
                        print(f"🔍 [API] 최종 grading_results: {grading_results}")
                    else:
                        print(f"⚠️ [API] 채점 결과 또는 사용자 답안이 없음")
                else:
                    print(f"⚠️ [API] score status가 success가 아님: {score_data.get('status')}")
            else:
                print(f"⚠️ [API] result에 score가 없음")
            
            return ChatResponse(
                response=response_text,
                service_used="et_agent",
                confidence=1.0,
                session_id=f"{request.user_id}:{request.chat_id}",
                grading_results=grading_results
            )
            
        except Exception as e:
            print(f"ET-Agent 실행 오류: {e}")
            # 오류 발생 시 PDF 생성 상태 초기화
            pdf_generation_status["is_generating"] = False
            pdf_generation_status["generated_files"] = []
            # 오류 발생 시 폴백 응답
            response_text = f"죄송합니다. ET-Agent 처리 중 오류가 발생했습니다: {str(e)}"
            return ChatResponse(
                response=response_text,
                service_used="error",
                confidence=0.0,
                session_id=f"{request.user_id}:{request.chat_id}"
            )
    
    # ET-Agent가 사용 불가능한 경우
    else:
        response_text = f"""
안녕하세요! ET-Agent입니다.

받은 메시지: {request.message}

현재 ET-Agent가 초기화되지 않았습니다.
서버를 재시작하거나 관리자에게 문의하세요.

농업 자격증 관련 질문을 해보세요!
        """.strip()
        
        return ChatResponse(
            response=response_text,
            service_used="fallback",
            confidence=0.5,
            session_id=f"{request.user_id}:{request.chat_id}"
        )

@app.post("/chat/teacher", response_model=ChatResponse)
async def chat_teacher(request: ChatRequest):
    """Teacher 서비스 시뮬레이션"""
    
    response_text = f"""
📚 Teacher 서비스 응답

질문: {request.message}

현재 Teacher 서비스는 개발 중입니다.
농업 자격증 관련 질문에 대한 기본 응답을 제공합니다.

예시 질문:
- "농산업기사 시험 문제를 만들어줘"
- "토양 관리에 대해 알려줘"
- "작물 병해충 방제 방법은?"
    """.strip()
    
    return ChatResponse(
        response=response_text,
        service_used="teacher",
        confidence=0.9,
        session_id=f"{request.user_id}:{request.chat_id}"
    )

@app.post("/chat/farmer", response_model=ChatResponse)
async def chat_farmer(request: ChatRequest):
    """Farmer 서비스 시뮬레이션"""
    
    response_text = f"""
🌱 Farmer 서비스 응답

질문: {request.message}

현재 Farmer 서비스는 개발 중입니다.
농업 관련 질문에 대한 기본 응답을 제공합니다.

예시 질문:
- "토마토 재배 방법은?"
- "적절한 작물을 추천해줘"
- "시기별 농작업은?"
    """.strip()
    
    return ChatResponse(
        response=response_text,
        service_used="farmer",
        confidence=0.9,
        session_id=f"{request.user_id}:{request.chat_id}"
    )

@app.get("/pdf-status")
async def get_pdf_status():
    """PDF 생성 상태 조회"""
    return {
        "is_generating": pdf_generation_status["is_generating"],
        "last_generated_time": pdf_generation_status["last_generated_time"],
        "generated_files": pdf_generation_status["generated_files"]
    }

@app.get("/pdfs")
async def list_pdfs():
    """최근 생성된 PDF 파일 목록 조회 (최근 1시간 내)"""
    try:
        import time
        current_time = time.time()
        one_hour_ago = current_time - 3600  # 1시간 전
        
        # PDF 파일들이 저장된 디렉토리들 확인
        pdf_dirs = [
            "teacher/agents/solution/pdf_outputs",
            "teacher/pdf_outputs", 
            "farmer/pdf_outputs"
        ]
        
        recent_pdf_files = []
        for pdf_dir in pdf_dirs:
            if os.path.exists(pdf_dir):
                files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
                for file_path in files:
                    file_created_time = os.path.getctime(file_path)
                    # 최근 1시간 내에 생성된 파일만 포함
                    if file_created_time >= one_hour_ago:
                        file_name = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path)
                        recent_pdf_files.append({
                            "filename": file_name,
                            "path": file_path,
                            "size": file_size,
                            "created": file_created_time
                        })
        
        # 생성 시간순으로 정렬 (최신순)
        recent_pdf_files.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "pdfs": recent_pdf_files,
            "count": len(recent_pdf_files),
            "filter": "최근 1시간 내 생성된 PDF만 표시"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 목록 조회 실패: {str(e)}")

@app.get("/recent-questions")
async def get_recent_questions(limit: int = 10):
    """최근 생성된 문제들 조회"""
    print(f"🔍 [API] /recent-questions 호출됨, limit={limit}")
    
    try:
        print(f"🔍 [API] orchestrator 상태 확인 중...")
        if not orchestrator:
            print("❌ [API] orchestrator가 None입니다!")
            return {"questions": [], "message": "MainOrchestrator가 초기화되지 않았습니다."}
        
        print(f"✅ [API] orchestrator 존재: {type(orchestrator).__name__}")
        
        # 디버깅 정보 추가
        debug_info = {
            "orchestrator_type": type(orchestrator).__name__,
            "has_memory": hasattr(orchestrator, 'memory'),
            "memory_type": type(orchestrator.memory).__name__ if hasattr(orchestrator, 'memory') else "None"
        }
        
        print(f"🔍 [API] 디버깅 정보: {debug_info}")
        
        print(f"🔍 [API] MainOrchestrator.load_recent_questions() 호출 중...")
        # MainOrchestrator에서 최근 문제들 불러오기
        recent_questions = orchestrator.load_recent_questions(limit=limit)
        print(f"🔍 [API] load_recent_questions 결과: {len(recent_questions)}개 문제")
        print(f"🔍 [API] recent_questions 상세: {recent_questions}")
        
        # Redis에서 불러온 문제가 없으면 빈 배열 유지
        if not recent_questions:
            print("📝 [API] Redis에 저장된 문제가 없습니다.")
            debug_info["redis_empty"] = True
        
        print(f"🔍 [API] 프론트엔드 형식으로 변환 시작...")
        # 프론트엔드에서 필요한 형태로 변환
        formatted_questions = []
        for i, q in enumerate(recent_questions, 1):
            print(f"🔍 [API] 문제 {i} 변환 중: {q.get('question', '')[:50]}...")
            formatted_question = {
                "id": i,
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "correctAnswer": q.get("answer", ""),
                "explanation": q.get("explanation", ""),
                "subject": q.get("subject", "unknown"),
                "created_at": q.get("created_at", 0)
            }
            formatted_questions.append(formatted_question)
            print(f"✅ [API] 문제 {i} 변환 완료: {formatted_question}")
        
        result = {
            "questions": formatted_questions,
            "count": len(formatted_questions),
            "message": f"최근 {len(formatted_questions)}개 문제를 불러왔습니다.",
            "debug": debug_info
        }
        
        print(f"✅ [API] 최종 결과: {result}")
        return result
        
    except Exception as e:
        print(f"❌ [API] 오류 발생: {str(e)}")
        import traceback
        print(f"❌ [API] 스택 트레이스: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"최근 문제 조회 실패: {str(e)}")

@app.post("/chat/clear")
async def clear_session(request: ChatRequest):
    """세션과 서비스 락 초기화"""
    try:
        if not orchestrator:
            return {
                "success": False,
                "message": "❌ ET-Agent 오케스트레이터가 초기화되지 않았습니다."
            }
        
        # Redis 메모리 초기화
        if hasattr(orchestrator, 'memory') and orchestrator.memory:
            try:
                # 완전한 세션 초기화 실행
                orchestrator.memory.clear_all_session_data()
                
                return {
                    "success": True,
                    "message": "✅ 세션과 서비스 락이 성공적으로 초기화되었습니다.",
                    "cleared_items": [
                        "세션 메모리 (shared, history)",
                        "문제 데이터",
                        "숏텀 메모리",
                        "채팅 히스토리",
                        "서비스 락",
                        "세션 관련 모든 키"
                    ],
                    "session_id": f"{request.user_id}:{request.chat_id}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"❌ 메모리 초기화 중 오류가 발생했습니다: {str(e)}"
                }
        else:
            return {
                "success": False,
                "message": "❌ Redis 메모리 인스턴스를 찾을 수 없습니다."
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ 초기화 중 오류가 발생했습니다: {str(e)}"
        }

# 간단한 clear 엔드포인트 (422 오류 방지)
@app.post("/clear")
async def simple_clear():
    """완전한 Redis 초기화 및 서비스 설정 리셋"""
    try:
        import redis
        
        # Redis 연결
        redis_client = redis.Redis(host="localhost", port=6380, decode_responses=True)
        
        # Redis 모든 데이터 삭제
        redis_client.flushall()
        print("🗑️ Redis 모든 데이터 삭제 완료")
        
        # ET-Agent 오케스트레이터 재초기화
        global orchestrator
        if orchestrator:
            try:
                # 기존 오케스트레이터 정리
                orchestrator = None
                print("🔄 기존 오케스트레이터 정리 완료")
            except Exception as e:
                print(f"⚠️ 기존 오케스트레이터 정리 중 오류: {e}")
        
        # 새로운 오케스트레이터 초기화
        try:
            print("🚀 새로운 ET-Agent 오케스트레이터 초기화 중...")
            orchestrator = MainOrchestrator(
                user_id="api_user",
                chat_id="api_chat"
            )
            print("✅ 새로운 ET-Agent 오케스트레이터 초기화 완료")
        except Exception as e:
            print(f"❌ 새로운 오케스트레이터 초기화 실패: {e}")
            orchestrator = None
        
        return {
            "success": True,
            "message": "✅ Redis 모든 데이터가 삭제되고 서비스가 완전히 초기화되었습니다.",
            "cleared_items": [
                "Redis 모든 데이터 (FLUSHALL)",
                "ET-Agent 오케스트레이터 재초기화",
                "세션 메모리 완전 삭제",
                "문제 데이터 완전 삭제",
                "숏텀 메모리 완전 삭제",
                "채팅 히스토리 완전 삭제",
                "서비스 락 완전 삭제",
                "모든 캐시 데이터 삭제"
            ]
        }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ 완전 초기화 중 오류가 발생했습니다: {str(e)}"
        }

@app.get("/pdf/{filename}")
async def download_pdf(filename: str):
    """PDF 파일 다운로드"""
    try:
        # PDF 파일들이 저장된 디렉토리들에서 파일 찾기
        pdf_dirs = [
            "teacher/agents/solution/pdf_outputs",
            "teacher/pdf_outputs", 
            "farmer/pdf_outputs"
        ]
        
        for pdf_dir in pdf_dirs:
            file_path = os.path.join(pdf_dir, filename)
            if os.path.exists(file_path):
                return FileResponse(
                    path=file_path,
                    filename=filename,
                    media_type="application/pdf"
                )
        
        raise HTTPException(status_code=404, detail="PDF 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 다운로드 실패: {str(e)}")

if __name__ == "__main__":
    print("🚀 ET-Agent Simple API 서버 시작")
    print("📍 주소: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("🔍 헬스 체크: http://localhost:8000/health")
    print("-" * 50)
    
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

