"use client";

// CopilotKit 의존성 제거
import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Question } from "./test/data/questions";

// AI 응답에서 문제를 파싱하는 함수
function parseQuestionsFromAIResponse(aiResponse: string, questionCount: number): Question[] {
  try {
    const questions: Question[] = [];
    
    // AI 응답에서 문제 패턴을 찾기
    const questionPattern = /문제\s*(\d+)[:：]\s*(.+?)(?=문제\s*\d+[:：]|$)/gs;
    const matches = [...aiResponse.matchAll(questionPattern)];
    
    for (let i = 0; i < Math.min(matches.length, questionCount); i++) {
      const match = matches[i];
      const questionText = match[2].trim();
      
      // 선택지 패턴 찾기 (A, B, C, D)
      const optionPattern = /([A-D])[:：]\s*([^\n]+)/g;
      const options: Record<string, string> = {};
      let optionMatch;
      
      while ((optionMatch = optionPattern.exec(questionText)) !== null) {
        options[optionMatch[1]] = optionMatch[2].trim();
      }
      
      // 정답 패턴 찾기
      const correctAnswerPattern = /정답[:：]\s*([A-D])/i;
      const correctMatch = questionText.match(correctAnswerPattern);
      const correctAnswer = correctMatch ? correctMatch[1] : 'A';
      
      // 해설 패턴 찾기
      const explanationPattern = /해설[:：]\s*(.+?)(?=문제\s*\d+[:：]|$)/s;
      const explanationMatch = questionText.match(explanationPattern);
      const explanation = explanationMatch ? explanationMatch[1].trim() : undefined;
      
      if (Object.keys(options).length >= 4) {
        questions.push({
          id: i + 1,
          question: questionText.split('\n')[0].trim(),
          options,
          correctAnswer,
          explanation,
          category: 'AI Generated',
          difficulty: 'Medium'
        });
      }
    }
    
    return questions;
  } catch (error) {
    console.error('AI 응답 파싱 오류:', error);
    return [];
  }
}

// 테스트 세션 타입 정의
type TestSession = {
  id: string;
  type: string;
  title: string;
  questions: Question[];
  currentQuestion: number;
  answers: Record<number, string>;
  startTime: Date;
  isCompleted: boolean;
  difficulty?: string;
  userLevel?: string;
  metadata?: any;
  analysis?: Record<number, any>;
  feedback?: any;
  recommendation?: any;
};

// 메인 컴포넌트를 동적으로 로드하여 SSR 완전 비활성화
const CopilotKitPage = dynamic(() => Promise.resolve(CopilotKitPageComponent), {
  ssr: false,
  loading: () => (
    <div className="h-screen w-screen bg-gray-200 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p className="text-gray-600">로딩 중...</p>
      </div>
    </div>
  ),
});

function CopilotKitPageComponent() {
  const [themeColor, setThemeColor] = useState("#6366f1");
  const [testSessions, setTestSessions] = useState<TestSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  // 테마 색상 변경 함수
  const handleThemeColorChange = (newColor: string) => {
    setThemeColor(newColor);
  };

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as React.CSSProperties}>
      <DynamicYourMainContent 
        themeColor={themeColor} 
        testSessions={testSessions}
        setTestSessions={setTestSessions}
        currentSessionId={currentSessionId}
        setCurrentSessionId={setCurrentSessionId}
      />
    </main>
  );
}

export default CopilotKitPage;

// 에이전트 상태는 필요시 다시 추가

// 동적 컴포넌트로 SSR 비활성화
const DynamicYourMainContent = dynamic(() => Promise.resolve(YourMainContent), {
  ssr: false,
  loading: () => (
    <div className="h-screen w-screen bg-gray-200 flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <p className="text-gray-600">로딩 중...</p>
      </div>
    </div>
  ),
});

function YourMainContent({ 
  themeColor, 
  testSessions, 
  setTestSessions, 
  currentSessionId, 
  setCurrentSessionId 
}: { 
  themeColor: string; 
  testSessions: TestSession[]; 
  setTestSessions: React.Dispatch<React.SetStateAction<TestSession[]>>; 
  currentSessionId: string | null; 
  setCurrentSessionId: React.Dispatch<React.SetStateAction<string | null>>; 
}) {
  // 에이전트 상태는 로컬 상태로 관리
  // useCoAgent는 제거하고 필요시 다시 추가

  // 테스트 세션 생성 함수 (AI 동적 생성)
  const createTestSession = async (type: string, questionCount: number, difficulty: string, userLevel: string) => {
    try {
      const title = `${type} ${questionCount}문제`;
      
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: `${type} ${questionCount}문제 만들어줘`,
          user_id: 'frontend_user',
          chat_id: 'frontend_chat'
        })
      });
      
      const data = await response.json();
      
      if (data.response) {
        // AI 응답에서 문제 파싱
        const aiQuestions = parseQuestionsFromAIResponse(data.response, questionCount);
        
        const sessionId = `session_${Math.random().toString(36).substr(2, 9)}`;
        const newSession: TestSession = {
          id: sessionId,
          type,
          title,
          questions: aiQuestions.length > 0 ? aiQuestions : [], // AI 생성 문제 사용
          currentQuestion: 0,
          answers: {},
          startTime: new Date(),
          isCompleted: false,
          difficulty,
          userLevel,
          metadata: { aiResponse: data.response }
        };
        
        setTestSessions(prev => [...prev, newSession]);
        setCurrentSessionId(sessionId);
      }
    } catch (error) {
      console.error('Failed to create test session:', error);
      // 폴백: 빈 문제 배열로 세션 생성
      const sessionId = `session_${Math.random().toString(36).substr(2, 9)}`;
      const newSession: TestSession = {
        id: sessionId,
        type,
        title: `${type} ${questionCount}문제`,
        questions: [], // 빈 배열로 시작
        currentQuestion: 0,
        answers: {},
        startTime: new Date(),
        isCompleted: false,
        difficulty,
        userLevel
      };
      
      setTestSessions(prev => [...prev, newSession]);
      setCurrentSessionId(sessionId);
    }
  };

  // 사용자 응답 분석 함수
  const analyzeUserResponse = async (sessionId: string, questionId: number, userAnswer: string, responseTime: number) => {
    try {
      const currentSession = testSessions.find(s => s.id === sessionId);
      if (!currentSession) return;
      
      const response = await fetch('/api/analyze-response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          questionId,
          userAnswer,
          responseTime,
          difficulty: currentSession.difficulty || 'intermediate'
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        // 분석 결과를 세션에 저장
        setTestSessions(prev => prev.map(s => 
          s.id === sessionId 
            ? { 
                ...s, 
                analysis: { 
                  ...s.analysis, 
                  [questionId]: data.analysis 
                },
                feedback: data.feedback,
                recommendation: data.recommendation
              }
            : s
        ));
        
        return data;
      }
    } catch (error) {
      console.error('Failed to analyze response:', error);
    }
  };

  // 날씨 카드 렌더링 함수
  const renderWeatherCard = (location: string) => {
    return <WeatherCard location={location} themeColor={themeColor} />;
  };

  return (
    <div
      className="h-screen w-screen bg-gray-200 flex transition-colors duration-300"
    >
            {/* 왼쪽: Proverbs 콘텐츠 또는 테스트 모드 */}
      <div className="flex-1 flex justify-center items-center p-8">
        {!currentSessionId ? (
          // 일반 모드 - 메인 페이지
          <div className="bg-gray-100/40 backdrop-blur-md p-8 rounded-2xl shadow-xl max-w-2xl w-full border border-gray-300">
            <div className="text-center">
              <p className="text-gray-600 text-sm">
                오른쪽 채팅창에서 "소프트웨어 설계 3문제 만들어줘"라고 입력하면<br/>
                AI가 생성한 문제 테스트를 시작할 수 있습니다!
              </p>
              {testSessions.length > 0 && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                  <p className="text-blue-800 text-sm">
                    최근 AI 응답: {testSessions[testSessions.length - 1]?.metadata?.aiResponse?.substring(0, 100)}...
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          // 테스트 모드 - 20문제 풀기 폼
          <div className="w-full h-full flex flex-col">
            <div className="text-center mb-6">
              <h1 className="text-3xl font-bold text-gray-800 mb-2">📝 정처기 문제 테스트</h1>
              <p className="text-gray-600">소프트웨어 설계 관련 20문제를 풀어보세요!</p>
            </div>
            
            {/* 문제 목록 - 전체 높이 사용 */}
            <div className="flex-1 overflow-y-auto pr-4">
              {currentSession && currentSession.questions.length > 0 ? (
                currentSession.questions.map((question, index) => (
                  <div key={question.id} className="mb-8 last:mb-0 bg-white rounded-lg p-6 shadow-sm">
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">
                      문제 {question.id}: {question.question}
                    </h3>
                    
                    {/* 보기 선택 */}
                    <div className="space-y-3">
                      {Object.entries(question.options).map(([key, value]) => (
                        <label key={key} className="flex items-center p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
                          <input 
                            type="radio" 
                            name={`q${question.id}`} 
                            value={key} 
                            className="mr-3" 
                          />
                          <span className="font-medium">{key}. {value}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <p className="text-gray-500">AI가 문제를 생성 중입니다...</p>
                  <p className="text-sm text-gray-400 mt-2">잠시만 기다려주세요.</p>
                </div>
              )}
            </div>
            
            {/* 전체 답안 제출 버튼 */}
            <div className="text-center mt-6">
              <button className="bg-green-500 hover:bg-green-600 text-white px-8 py-4 rounded-lg font-medium transition-colors shadow-lg text-lg mr-4">
                전체 답안 제출하기
              </button>
              <button 
                onClick={() => setCurrentSessionId(null)}
                className="bg-gray-500 hover:bg-gray-600 text-white px-6 py-3 rounded-lg font-medium transition-colors shadow-lg"
              >
                메인으로 돌아가기
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 오른쪽: 채팅 영역 */}
      <div className="w-[500px] bg-white shadow-xl p-8 relative">
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold text-gray-800 mb-4">👋 안녕하세요! 이장님과 선생님입니다.</h3>
          <p className="text-gray-600 mb-6">이 에이전트는 사용자의 질문을 파악하여 정처기 및 농사관련 질문에 답변을 할수있습니다.</p>
          
          <div className="text-left">
            <p className="text-gray-700 font-medium mb-3">예를 들어 다음과 같이 시도해볼 수 있습니다:</p>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">•</span>
                <span><strong>농사 질문:</strong> "오이에는 어떤 병해충이 있어?"</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">•</span>
                <span><strong>정보처리기사 질문:</strong> "소프트웨어 설계 3문제 만들어줘"</span>
              </li>
            </ul>
          </div>
          
          <p className="text-gray-600 text-sm mt-6">
            질문을 입력하시면 질문에 따라 농사관련 질문이나 정처기 관련 질문을 할수있습니다.
          </p>
        </div>
        
                 {/* 실제 채팅 영역 */}
         <div className="absolute bottom-0 left-0 right-0 p-6">
           <ChatInterface />
         </div>
      </div>
    </div>
  );
}

// 채팅 인터페이스 컴포넌트
function ChatInterface() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [availablePdfs, setAvailablePdfs] = useState<Array<{filename: string, size: number, created: number}>>([]);
  const [isLoadingPdfs, setIsLoadingPdfs] = useState(false);
  const [pdfGenerationStatus, setPdfGenerationStatus] = useState<{
    is_generating: boolean;
    last_generated_time: number | null;
    generated_files: string[];
  }>({
    is_generating: false,
    last_generated_time: null,
    generated_files: []
  });

  // 백엔드 연결 테스트 함수
  const testBackendConnection = async () => {
    setIsTestingConnection(true);
    try {
      const response = await fetch("http://localhost:8000/health", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setIsBackendConnected(true);
        console.log("✅ 백엔드 연결 성공:", data);
      } else {
        setIsBackendConnected(false);
        console.error("❌ 백엔드 연결 실패:", response.status);
      }
    } catch (error) {
      setIsBackendConnected(false);
      console.error("❌ 백엔드 연결 오류:", error);
    } finally {
      setIsTestingConnection(false);
    }
  };

  // PDF 생성 상태 확인 함수
  const checkPdfGenerationStatus = async () => {
    try {
      const response = await fetch("http://localhost:8000/pdf-status", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setPdfGenerationStatus(data);
        
        // 새로 생성된 PDF가 있으면 목록 새로고침
        if (data.generated_files && data.generated_files.length > 0) {
          fetchPdfs();
        }
        
        console.log("✅ PDF 생성 상태 확인:", data);
      } else {
        console.error("❌ PDF 생성 상태 확인 실패:", response.status);
      }
    } catch (error) {
      console.error("❌ PDF 생성 상태 확인 오류:", error);
    }
  };

  // PDF 목록 조회 함수
  const fetchPdfs = async () => {
    setIsLoadingPdfs(true);
    try {
      const response = await fetch("http://localhost:8000/pdfs", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setAvailablePdfs(data.pdfs || []);
        console.log("✅ PDF 목록 조회 성공:", data);
      } else {
        console.error("❌ PDF 목록 조회 실패:", response.status);
      }
    } catch (error) {
      console.error("❌ PDF 목록 조회 오류:", error);
    } finally {
      setIsLoadingPdfs(false);
    }
  };

  // PDF 다운로드 함수
  const downloadPdf = async (filename: string) => {
    try {
      const response = await fetch(`http://localhost:8000/pdf/${filename}`, {
        method: "GET",
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        console.log("✅ PDF 다운로드 성공:", filename);
      } else {
        console.error("❌ PDF 다운로드 실패:", response.status);
      }
    } catch (error) {
      console.error("❌ PDF 다운로드 오류:", error);
    }
  };

  // 컴포넌트 마운트 시 자동으로 연결 테스트 및 PDF 생성 상태 확인
  useEffect(() => {
    testBackendConnection();
    checkPdfGenerationStatus();
  }, []);

  const sendMessage = async () => {
    if (!message.trim()) return;
    
    const userMessage = { role: "user", content: message };
    setMessages(prev => [...prev, userMessage]);
    setMessage("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          user_id: "frontend_user",
          chat_id: "frontend_chat",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage = { role: "assistant", content: data.response || "응답을 생성할 수 없습니다." };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("백엔드 API 호출 실패:", error);
      const errorMessage = { role: "assistant", content: "죄송합니다. 백엔드 서버에 연결할 수 없습니다." };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      // 메시지 전송 후 PDF 생성 상태 체크 (여러 번 체크)
      setTimeout(() => checkPdfGenerationStatus(), 1000);  // 1초 후 첫 번째 체크
      setTimeout(() => checkPdfGenerationStatus(), 3000);  // 3초 후 두 번째 체크
      setTimeout(() => checkPdfGenerationStatus(), 5000);  // 5초 후 세 번째 체크
      setTimeout(() => checkPdfGenerationStatus(), 10000); // 10초 후 마지막 체크
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg h-96 flex flex-col">
      {/* 채팅 헤더 */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h4 className="text-lg font-semibold text-gray-800">ET-Agent 채팅</h4>
          <div className="flex items-center space-x-2">
            {/* 연결 상태 표시 */}
            {isBackendConnected === null ? (
              <div className="flex items-center space-x-1">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-gray-600"></div>
                <span className="text-xs text-gray-500">연결 확인 중...</span>
              </div>
            ) : isBackendConnected ? (
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-xs text-green-600">연결됨</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                <span className="text-xs text-red-600">연결 안됨</span>
              </div>
            )}
            <button
              onClick={checkPdfGenerationStatus}
              disabled={isLoadingPdfs}
              className="text-xs px-2 py-1 bg-blue-100 hover:bg-blue-200 rounded text-blue-600 disabled:opacity-50"
            >
              {isLoadingPdfs ? "확인 중..." : "PDF 상태 확인"}
            </button>
            <button
              onClick={testBackendConnection}
              disabled={isTestingConnection}
              className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-600 disabled:opacity-50"
            >
              {isTestingConnection ? "테스트 중..." : "재연결"}
            </button>
          </div>
        </div>
      </div>
      
      {/* 연결 상태에 따른 내용 표시 */}
      {isBackendConnected === false ? (
        // 연결 실패 시 화면
        <div className="flex-1 flex items-center justify-center p-4">
          <div className="text-center">
            <div className="text-red-500 text-4xl mb-4">⚠️</div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">백엔드 서버 연결 실패</h3>
            <p className="text-sm text-gray-600 mb-4">
              백엔드 서버가 실행되지 않았거나 연결할 수 없습니다.
            </p>
            <div className="space-y-2 text-xs text-gray-500">
              <p>• 백엔드 서버가 실행 중인지 확인하세요</p>
              <p>• <code className="bg-gray-100 px-1 rounded">http://localhost:8000</code>에 접근 가능한지 확인하세요</p>
            </div>
            <button
              onClick={testBackendConnection}
              disabled={isTestingConnection}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {isTestingConnection ? "연결 테스트 중..." : "다시 연결 시도"}
            </button>
          </div>
        </div>
      ) : (
        // 연결 성공 시 채팅 화면
        <>
          {/* PDF 생성 상태 표시 */}
          {pdfGenerationStatus.is_generating && (
            <div className="border-b border-gray-200 p-3 bg-yellow-50">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600"></div>
                <span className="text-sm text-yellow-700">PDF 생성 중...</span>
              </div>
            </div>
          )}
          
          {/* PDF 다운로드 섹션 */}
          {availablePdfs.length > 0 && (
            <div className="border-b border-gray-200 p-3 bg-blue-50">
              <div className="flex items-center justify-between mb-2">
                <h5 className="text-sm font-medium text-blue-700">📄 새로 생성된 PDF 파일</h5>
                <span className="text-xs text-blue-500">{availablePdfs.length}개</span>
              </div>
              <p className="text-xs text-blue-600 mb-2">백엔드에서 PDF 생성 완료 시 자동으로 표시됩니다</p>
              <div className="space-y-1 max-h-20 overflow-y-auto">
                {availablePdfs.slice(0, 3).map((pdf, index) => (
                  <div key={index} className="flex items-center justify-between bg-white rounded px-2 py-1 text-xs">
                    <span className="text-gray-600 truncate flex-1 mr-2" title={pdf.filename}>
                      {pdf.filename}
                    </span>
                    <button
                      onClick={() => downloadPdf(pdf.filename)}
                      className="px-2 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-xs"
                    >
                      다운로드
                    </button>
                  </div>
                ))}
                {availablePdfs.length > 3 && (
                  <div className="text-xs text-gray-500 text-center">
                    +{availablePdfs.length - 3}개 더...
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* 메시지 영역 */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500">
                <p>안녕하세요! 질문을 입력해주세요.</p>
                <p className="text-sm mt-2">예: "소프트웨어 설계 3문제 만들어줘"</p>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      msg.role === "user"
                        ? "bg-blue-500 text-white"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-800 px-4 py-2 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                    <span className="text-sm">응답 중...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* 입력 영역 */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex space-x-2">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="질문을 입력하세요..."
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isLoading || isBackendConnected !== true}
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !message.trim() || isBackendConnected !== true}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                전송
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// Simple sun icon for the weather card
function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-14 h-14 text-yellow-200">
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" strokeWidth="2" stroke="currentColor" />
    </svg>
  );
}

// Weather card component where the location and themeColor are based on what the agent
// sets via tool calls.
function WeatherCard({ location, themeColor }: { location?: string, themeColor: string }) {
  return (
    <div
    style={{ backgroundColor: themeColor }}
    className="rounded-xl shadow-xl mt-6 mb-4 max-w-md w-full"
  >
    <div className="bg-white/20 p-4 w-full">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-white capitalize">{location}</h3>
          <p className="text-white">Current Weather</p>
        </div>
        <SunIcon />
      </div>
      
      <div className="mt-4 flex items-end justify-between">
        <div className="text-3xl font-bold text-white">70°</div>
        <div className="text-sm text-white">Clear skies</div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-white">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-white text-xs">Humidity</p>
            <p className="text-white font-medium">45%</p>
          </div>
          <div>
            <p className="text-white text-xs">Wind</p>
            <p className="text-white font-medium">5 mph</p>
          </div>
          <div>
            <p className="text-white text-xs">Feels Like</p>
            <p className="text-white font-medium">72°</p>
          </div>
        </div>
      </div>
    </div>
  </div>
  );
}
