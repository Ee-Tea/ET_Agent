"""
헬스 체크 및 시스템 상태 API 라우터
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import psutil
import time

from ..models import HealthResponse, SystemStatsResponse, ServiceStats

router = APIRouter(prefix="/health", tags=["health"])

# 서비스 시작 시간
start_time = datetime.now()

# 서비스 통계
service_stats = {
    "orchestrator": ServiceStats(
        service_name="orchestrator",
        total_requests=0,
        successful_requests=0,
        failed_requests=0,
        average_response_time=0.0
    ),
    "teacher": ServiceStats(
        service_name="teacher",
        total_requests=0,
        successful_requests=0,
        failed_requests=0,
        average_response_time=0.0
    ),
    "farmer": ServiceStats(
        service_name="farmer",
        total_requests=0,
        successful_requests=0,
        failed_requests=0,
        average_response_time=0.0
    )
}

# 전역 변수
orchestrator = None
teacher = None

def set_services(orch, teach):
    """서비스 인스턴스 설정"""
    global orchestrator, teacher
    orchestrator = orch
    teacher = teach

def update_service_stats(service_name: str, success: bool, response_time: float):
    """서비스 통계 업데이트"""
    if service_name in service_stats:
        stats = service_stats[service_name]
        stats.total_requests += 1
        
        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
        
        # 평균 응답 시간 계산 (간단한 이동 평균)
        if stats.average_response_time == 0:
            stats.average_response_time = response_time
        else:
            stats.average_response_time = (stats.average_response_time + response_time) / 2
        
        stats.last_request_time = datetime.now().isoformat()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """기본 헬스 체크"""
    try:
        # 각 서비스 상태 확인
        services = {
            "orchestrator": "healthy" if orchestrator else "unhealthy",
            "teacher": "healthy" if teacher else "unhealthy",
            "redis": "unknown",  # Redis 연결 상태는 별도로 확인 필요
            "api": "healthy"
        }
        
        overall_status = "healthy" if all(
            status == "healthy" for status in services.values() 
            if status != "unknown"
        ) else "unhealthy"
        
        # 가동 시간 계산
        uptime = datetime.now() - start_time
        uptime_str = str(uptime).split('.')[0]  # 마이크로초 제거
        
        return HealthResponse(
            status=overall_status,
            version="1.0.0",
            services=services,
            uptime=uptime_str
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/detailed")
async def detailed_health_check():
    """상세 헬스 체크"""
    try:
        # 시스템 리소스 정보
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 서비스별 상세 상태
        service_details = {}
        
        # Orchestrator 상태
        if orchestrator:
            service_details["orchestrator"] = {
                "status": "healthy",
                "initialized": True,
                "memory_available": hasattr(orchestrator, 'memory'),
                "checkpointer_available": hasattr(orchestrator, 'checkpointer')
            }
        else:
            service_details["orchestrator"] = {
                "status": "unhealthy",
                "initialized": False
            }
        
        # Teacher 상태
        if teacher:
            service_details["teacher"] = {
                "status": "healthy",
                "initialized": True,
                "agents_loaded": hasattr(teacher, 'retriever_runner') and teacher.retriever_runner is not None
            }
        else:
            service_details["teacher"] = {
                "status": "unhealthy",
                "initialized": False
            }
        
        # Redis 연결 상태 (간단한 체크)
        try:
            if orchestrator and hasattr(orchestrator, 'memory'):
                # Redis 연결 테스트
                service_details["redis"] = {
                    "status": "healthy",
                    "connected": True
                }
            else:
                service_details["redis"] = {
                    "status": "unknown",
                    "connected": False
                }
        except Exception:
            service_details["redis"] = {
                "status": "unhealthy",
                "connected": False
            }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - start_time).split('.')[0],
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "services": service_details
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {str(e)}")

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """시스템 통계 조회"""
    try:
        # 가동 시간
        uptime = datetime.now() - start_time
        uptime_str = str(uptime).split('.')[0]
        
        # 메모리 사용량
        memory = psutil.virtual_memory()
        memory_usage = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # 총 세션 수 (임시)
        total_sessions = 10  # 실제로는 DB에서 조회
        active_sessions = 5  # 실제로는 활성 세션 수 조회
        
        # 총 메시지 수
        total_messages = sum(stats.total_requests for stats in service_stats.values())
        
        return SystemStatsResponse(
            total_sessions=total_sessions,
            active_sessions=active_sessions,
            total_messages=total_messages,
            services=list(service_stats.values()),
            uptime=uptime_str,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@router.get("/services/{service_name}")
async def get_service_health(service_name: str):
    """특정 서비스 헬스 체크"""
    try:
        if service_name not in service_stats:
            raise HTTPException(status_code=404, detail="Service not found")
        
        stats = service_stats[service_name]
        
        # 서비스별 상세 상태
        service_status = {
            "service_name": service_name,
            "status": "healthy",
            "stats": stats.dict(),
            "last_check": datetime.now().isoformat()
        }
        
        # 서비스별 특별 체크
        if service_name == "orchestrator":
            service_status["initialized"] = orchestrator is not None
        elif service_name == "teacher":
            service_status["initialized"] = teacher is not None
            if teacher:
                service_status["agents_loaded"] = hasattr(teacher, 'retriever_runner')
        
        return service_status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service health check failed: {str(e)}")

@router.post("/services/{service_name}/test")
async def test_service(service_name: str):
    """서비스 테스트"""
    try:
        start_time_test = time.time()
        success = False
        
        if service_name == "orchestrator":
            if orchestrator:
                # 간단한 테스트 실행
                test_result = orchestrator.run("테스트", config={
                    "configurable": {"thread_id": "test"}
                })
                success = test_result is not None
            else:
                success = False
                
        elif service_name == "teacher":
            if teacher:
                # 간단한 테스트 실행
                test_state = {
                    "user_query": "테스트",
                    "intent": "",
                    "shared": {},
                    "work": {},
                    "retrieval": {},
                    "generation": {},
                    "solution": {},
                    "score": {},
                    "analysis": {},
                    "history": [],
                    "session": {},
                    "artifacts": {},
                    "routing": {},
                    "llm_response": ""
                }
                test_result = teacher.execute(test_state)
                success = test_result is not None
            else:
                success = False
        else:
            raise HTTPException(status_code=404, detail="Service not found")
        
        response_time = time.time() - start_time_test
        
        # 통계 업데이트
        update_service_stats(service_name, success, response_time)
        
        return {
            "service_name": service_name,
            "test_result": "success" if success else "failed",
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service test failed: {str(e)}")


