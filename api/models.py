"""
FastAPI Pydantic 모델 정의
API 요청/응답 스키마
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

# ========== 기본 요청/응답 모델 ==========

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=2000)
    user_id: str = Field(default="api_user", description="사용자 ID")
    chat_id: str = Field(default="api_chat", description="채팅 ID")
    service_type: Optional[str] = Field(default=None, description="서비스 타입 (teacher/farmer)")
    session_id: Optional[str] = Field(default=None, description="세션 ID")

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str = Field(..., description="에이전트 응답")
    service_used: str = Field(..., description="사용된 서비스")
    confidence: float = Field(..., description="응답 신뢰도")
    session_id: str = Field(..., description="세션 ID")
    artifacts: Optional[Dict[str, Any]] = Field(default=None, description="생성된 파일들")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="응답 시간")

class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str = Field(..., description="서비스 상태")
    version: str = Field(..., description="API 버전")
    services: Dict[str, str] = Field(..., description="서비스별 상태")
    uptime: Optional[str] = Field(default=None, description="서비스 가동 시간")

# ========== 세션 관리 모델 ==========

class SessionRequest(BaseModel):
    """세션 생성 요청 모델"""
    user_id: str = Field(..., description="사용자 ID")
    chat_id: Optional[str] = Field(default=None, description="채팅 ID (자동 생성 가능)")
    service_type: str = Field(default="teacher", description="서비스 타입")

class SessionResponse(BaseModel):
    """세션 응답 모델"""
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    chat_id: str = Field(..., description="채팅 ID")
    title: Optional[str] = Field(default=None, description="세션 제목")
    created_at: str = Field(..., description="생성 시간")
    status: str = Field(..., description="세션 상태")
    service_type: str = Field(..., description="서비스 타입")

class SessionListResponse(BaseModel):
    """세션 목록 응답 모델"""
    sessions: List[SessionResponse] = Field(..., description="세션 목록")
    total: int = Field(..., description="총 세션 수")

# ========== Teacher 서비스 모델 ==========

class TeacherRequest(BaseModel):
    """Teacher 서비스 요청 모델"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=2000)
    user_id: str = Field(default="api_user", description="사용자 ID")
    chat_id: str = Field(default="api_chat", description="채팅 ID")
    intent: Optional[str] = Field(default=None, description="의도 분류")
    output_mode: Optional[str] = Field(default="text", description="출력 모드 (text/pdf/form)")

class TeacherResponse(BaseModel):
    """Teacher 서비스 응답 모델"""
    response: str = Field(..., description="에이전트 응답")
    intent: str = Field(..., description="분류된 의도")
    artifacts: Optional[Dict[str, Any]] = Field(default=None, description="생성된 파일들")
    session_id: str = Field(..., description="세션 ID")
    confidence: float = Field(default=1.0, description="응답 신뢰도")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="응답 시간")

# ========== Farmer 서비스 모델 ==========

class FarmerRequest(BaseModel):
    """Farmer 서비스 요청 모델"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=2000)
    user_id: str = Field(default="api_user", description="사용자 ID")
    chat_id: str = Field(default="api_chat", description="채팅 ID")
    crop_type: Optional[str] = Field(default=None, description="작물 종류")
    region: Optional[str] = Field(default=None, description="지역")

class FarmerResponse(BaseModel):
    """Farmer 서비스 응답 모델"""
    response: str = Field(..., description="에이전트 응답")
    recommendations: Optional[List[str]] = Field(default=None, description="추천 사항")
    session_id: str = Field(..., description="세션 ID")
    confidence: float = Field(default=1.0, description="응답 신뢰도")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="응답 시간")

# ========== 파일 업로드 모델 ==========

class FileUploadRequest(BaseModel):
    """파일 업로드 요청 모델"""
    user_id: str = Field(..., description="사용자 ID")
    chat_id: str = Field(..., description="채팅 ID")
    file_type: str = Field(..., description="파일 타입 (pdf/image)")
    description: Optional[str] = Field(default=None, description="파일 설명")

class FileUploadResponse(BaseModel):
    """파일 업로드 응답 모델"""
    file_id: str = Field(..., description="파일 ID")
    filename: str = Field(..., description="파일명")
    file_type: str = Field(..., description="파일 타입")
    size: int = Field(..., description="파일 크기 (bytes)")
    upload_time: str = Field(default_factory=lambda: datetime.now().isoformat(), description="업로드 시간")
    status: str = Field(..., description="처리 상태")

# ========== 히스토리 모델 ==========

class ChatHistoryItem(BaseModel):
    """채팅 히스토리 아이템"""
    timestamp: str = Field(..., description="시간")
    user_message: str = Field(..., description="사용자 메시지")
    bot_response: str = Field(..., description="봇 응답")
    service_used: str = Field(..., description="사용된 서비스")
    confidence: float = Field(..., description="신뢰도")

class ChatHistoryResponse(BaseModel):
    """채팅 히스토리 응답 모델"""
    session_id: str = Field(..., description="세션 ID")
    history: List[ChatHistoryItem] = Field(..., description="채팅 히스토리")
    total_count: int = Field(..., description="총 메시지 수")
    start_date: Optional[str] = Field(default=None, description="시작 날짜")
    end_date: Optional[str] = Field(default=None, description="종료 날짜")

# ========== 에러 모델 ==========

class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(default=None, description="상세 에러 정보")
    status_code: int = Field(..., description="HTTP 상태 코드")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="에러 발생 시간")
    path: str = Field(..., description="요청 경로")

# ========== 통계 모델 ==========

class ServiceStats(BaseModel):
    """서비스 통계 모델"""
    service_name: str = Field(..., description="서비스명")
    total_requests: int = Field(..., description="총 요청 수")
    successful_requests: int = Field(..., description="성공한 요청 수")
    failed_requests: int = Field(..., description="실패한 요청 수")
    average_response_time: float = Field(..., description="평균 응답 시간 (초)")
    last_request_time: Optional[str] = Field(default=None, description="마지막 요청 시간")

class SystemStatsResponse(BaseModel):
    """시스템 통계 응답 모델"""
    total_sessions: int = Field(..., description="총 세션 수")
    active_sessions: int = Field(..., description="활성 세션 수")
    total_messages: int = Field(..., description="총 메시지 수")
    services: List[ServiceStats] = Field(..., description="서비스별 통계")
    uptime: str = Field(..., description="서비스 가동 시간")
    memory_usage: Optional[Dict[str, Any]] = Field(default=None, description="메모리 사용량")
