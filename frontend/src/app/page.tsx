"use client";

// CopilotKit 의존성 제거
import { useState, useEffect } from "react";
import dynamic from "next/dynamic";

// 테스트 세션 타입 정의
type TestSession = {
  id: string;
  type: string;
  title: string;
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
  
  // 문제 관련 상태
  const [parsedQuestions, setParsedQuestions] = useState<Array<{
    id: number;
    question: string;
    options: string[];
    correctAnswer: string;
    explanation: string;
    subject: string;
    created_at: number;
  }>>([]);
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(false);
  const [selectedAnswers, setSelectedAnswers] = useState<{[key: string]: string}>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [gradingResults, setGradingResults] = useState<{[key: string]: any}>({});
  const [pdfGenerationStatus, setPdfGenerationStatus] = useState({
    is_generating: false,
    last_generated_time: null,
    generated_files: []
  });
  const [availablePdfs, setAvailablePdfs] = useState<any[]>([]);

  // gradingResults 상태 변경 감지
  useEffect(() => {
    console.log("🔍 gradingResults 상태 변경:", gradingResults);
  }, [gradingResults]);

  // PDF 관련 함수들
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

  const fetchPdfs = async () => {
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
    }
  };

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

  // 컴포넌트 마운트 시 PDF 상태 확인
  useEffect(() => {
    checkPdfGenerationStatus();
  }, []);

  // 문제 선택 핸들러
  const handleAnswerSelect = (questionId: string, answer: string) => {
    setSelectedAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
  };

  // 모든 문제가 답변되었는지 확인
  const isAllQuestionsAnswered = () => {
    return parsedQuestions.length > 0 && parsedQuestions.every(q => selectedAnswers[`question-${q.id}`]);
  };

  // 답안 제출 함수
  const submitAnswers = async () => {
    if (!isAllQuestionsAnswered()) return;
    
    setIsSubmitting(true);
    try {
      // 답안을 숫자로 변환하여 쿼리 생성
      const answers = parsedQuestions.map(q => {
        const questionId = `question-${q.id}`;
        const selectedAnswer = selectedAnswers[questionId];
        
        // 선택된 답안에서 숫자만 추출 (예: "  1. 코드 재사용성 향상" -> "1")
        const extractedAnswer = selectedAnswer ? selectedAnswer.match(/^\s*(\d+)\./)?.[1] : null;
        
        console.log(`🔍 문제 ${q.id} 답안 변환:`, {
          selectedAnswer,
          extractedAnswer,
          questionId
        });
        
        return extractedAnswer || "1"; // 추출 실패 시 기본값 1
      });
      
      const query = `${answers.join(',')} + 문제의 답이야 채점해줘`;
      
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: query,
          user_id: "frontend_user",
          chat_id: "frontend_chat",
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("🔍 채점 결과 전체 응답:", data);
        console.log("🔍 grading_results 존재 여부:", !!data.grading_results);
        console.log("🔍 grading_results 내용:", data.grading_results);
        
        // 백엔드에서 받은 채점 결과 사용
        if (data.grading_results) {
          console.log("✅ 백엔드 채점 결과 사용:", data.grading_results);
          setGradingResults(data.grading_results);
        } else {
          console.log("⚠️ 백엔드 채점 결과 없음, 프론트엔드에서 계산");
          
          // 백엔드에서 채점 결과가 없는 경우 프론트엔드에서 간단히 비교
          const newGradingResults: {[key: string]: any} = {};
          console.log("🔍 문제 수:", parsedQuestions.length);
          console.log("🔍 선택된 답안들:", selectedAnswers);
          
          parsedQuestions.forEach((q, index) => {
            const questionId = `question-${q.id}`;
            const selectedAnswer = selectedAnswers[questionId];
            const correctAnswer = q.correctAnswer;
            
            // 사용자 답안에서 숫자만 추출 (예: "  1. 코드 재사용성 향상" -> "1")
            const extractedUserAnswer = selectedAnswer ? selectedAnswer.match(/^\s*(\d+)\./)?.[1] : null;
            
            console.log(`🔍 문제 ${q.id}:`, {
              questionId,
              selectedAnswer,
              extractedUserAnswer,
              correctAnswer,
              isCorrect: extractedUserAnswer === correctAnswer
            });
            
            newGradingResults[questionId] = {
              isCorrect: extractedUserAnswer === correctAnswer,
              userAnswer: selectedAnswer,
              correctAnswer: correctAnswer
            };
          });
          
          setGradingResults(newGradingResults);
          console.log("🔍 최종 프론트엔드 채점 결과:", newGradingResults);
        }
      }
    } catch (error) {
      console.error("답안 제출 실패:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

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
        
        const sessionId = `session_${Math.random().toString(36).substr(2, 9)}`;
        const newSession: TestSession = {
          id: sessionId,
          type,
          title,
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

  // 최근 생성된 문제들을 불러오는 함수
  const fetchRecentQuestions = async () => {
    setIsLoadingQuestions(true);
    try {
      const response = await fetch(`http://localhost:8000/recent-questions`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setParsedQuestions(data.questions || []);
        console.log("✅ 최근 문제 조회 성공:", data);
      } else {
        console.error("❌ 최근 문제 조회 실패:", response.status);
      }
    } catch (error) {
      console.error("❌ 최근 문제 조회 오류:", error);
      // 폴백: 빈 배열로 설정
      setParsedQuestions([]);
    } finally {
      setIsLoadingQuestions(false);
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

  // 컴포넌트 마운트 시 최근 문제들 불러오기
  useEffect(() => {
    fetchRecentQuestions();
  }, []);

  // 날씨 카드 렌더링 함수
  const renderWeatherCard = (location: string) => {
    return <WeatherCard location={location} themeColor={themeColor} />;
  };

  return (
    <div
      className="h-screen w-screen bg-gray-200 flex transition-colors duration-300"
    >
      {/* 왼쪽: 문제 표시 영역 */}
      <div className="w-3/5 flex flex-col p-4">
        <div className="bg-white rounded-lg shadow-lg h-full flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">📝 최근 생성된 문제</h3>
                <p className="text-sm text-gray-600">AI가 생성한 최신 문제들을 확인하세요</p>
              </div>
              
              {/* PDF 다운로드 섹션 */}
              <div className="flex flex-col items-end space-y-2">
                {/* PDF 생성 상태 표시 */}
                {pdfGenerationStatus.is_generating && (
                  <div className="flex items-center space-x-2 px-3 py-1 bg-yellow-50 rounded-lg">
                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-yellow-600"></div>
                    <span className="text-xs text-yellow-700">PDF 생성 중...</span>
                  </div>
                )}
                
                {/* PDF 다운로드 섹션 */}
                {availablePdfs.length > 0 && (
                  <div className="bg-blue-50 rounded-lg p-3 min-w-64">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="text-sm font-medium text-blue-700">📄 새로 생성된 PDF</h5>
                      <span className="text-xs text-blue-500">{availablePdfs.length}개</span>
                    </div>
                    <div className="space-y-1 max-h-24 overflow-y-auto">
                      {availablePdfs.slice(0, 2).map((pdf, index) => (
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
                      {availablePdfs.length > 2 && (
                        <div className="text-xs text-gray-500 text-center">
                          +{availablePdfs.length - 2}개 더...
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {isLoadingQuestions ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                <p className="text-gray-600">문제를 파싱하는 중...</p>
              </div>
            </div>
          ) : parsedQuestions.length > 0 ? (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {parsedQuestions.map((question, questionIndex) => {
                const questionId = `question-${question.id}`;
                return (
                  <div key={question.id} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                    <div className="mb-3">
                      <span className="inline-block bg-blue-100 text-blue-800 text-xs font-medium px-2 py-1 rounded-full mr-2">
                        문제 {question.id}
                      </span>
                      <p className="text-gray-800 font-medium mt-2">{question.question}</p>
                    </div>
                    
                    <div className="space-y-2">
                      {Array.isArray(question.options) && question.options.map((option, index) => (
                        <label key={index} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-100 p-2 rounded">
                          <input
                            type="radio"
                            name={questionId}
                            value={option}
                            checked={selectedAnswers[questionId] === option}
                            onChange={() => handleAnswerSelect(questionId, option)}
                            className="text-blue-600 focus:ring-blue-500"
                          />
                          <span className="text-sm text-gray-700">{option}</span>
                        </label>
                      ))}
                      {!Array.isArray(question.options) && (
                        <p className="text-sm text-gray-500">보기 정보가 없습니다.</p>
                      )}
                    </div>
                    
                    {/* 채점 결과 표시 */}
                    {(() => {
                      const gradingResult = gradingResults[questionId];
                      console.log(`🔍 문제 ${question.id} 채점 결과:`, {
                        questionId,
                        gradingResult,
                        hasGradingResult: !!gradingResult,
                        isCorrect: gradingResult?.isCorrect,
                        userAnswer: gradingResult?.userAnswer,
                        correctAnswer: question.correctAnswer
                      });
                      
                      return gradingResult && (
                        <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded">
                          <div className="flex items-center space-x-2 mb-2">
                            <span className={`text-sm font-medium ${
                              gradingResult.isCorrect ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {gradingResult.isCorrect ? '✅ 정답' : '❌ 오답'}
                            </span>
                            <span className="text-sm text-gray-600">
                              정답: {question.correctAnswer}
                            </span>
                          </div>
                          {question.explanation && (
                            <p className="text-sm text-gray-700">
                              <span className="font-medium">해설:</span> {question.explanation}
                            </p>
                          )}
                        </div>
                      );
                    })()}
                    
                  </div>
                );
              })}
              
              {/* 제출 버튼 */}
              {parsedQuestions.length > 0 && (
                <div className="mt-4 p-4 border-t border-gray-200">
                  <button
                    onClick={submitAnswers}
                    disabled={!isAllQuestionsAnswered() || isSubmitting}
                    className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                      isAllQuestionsAnswered() && !isSubmitting
                        ? 'bg-blue-500 hover:bg-blue-600 text-white'
                        : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    }`}
                  >
                    {isSubmitting ? (
                      <div className="flex items-center justify-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        <span>채점 중...</span>
                      </div>
                    ) : (
                      `답안 제출하기 (${Object.keys(selectedAnswers).length}/${parsedQuestions.length})`
                    )}
                  </button>
                  {!isAllQuestionsAnswered() && (
                    <p className="text-sm text-gray-500 mt-2 text-center">
                      모든 문제에 답변해주세요
                    </p>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-gray-500">
                <p className="text-lg mb-2">📝</p>
                <p>아직 생성된 문제가 없습니다</p>
                <p className="text-sm mt-1">채팅에서 "소프트웨어 설계 3문제 만들어줘"라고 입력해보세요</p>
              </div>
            </div>
          )}
        </div>
      </div>



      {/* 오른쪽: 채팅 영역 */}
      <div className="w-2/5 bg-white shadow-xl p-8 relative">
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
           <ChatInterface onMessageSent={fetchRecentQuestions} />
         </div>
      </div>
    </div>
  );
}

// 채팅 인터페이스 컴포넌트
function ChatInterface({ onMessageSent }: { onMessageSent: () => void }) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean | null>(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);

  // 최근 생성된 문제들을 불러오는 함수

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

  // 컴포넌트 마운트 시 자동으로 연결 테스트
  useEffect(() => {
    testBackendConnection();
  }, []);

  const sendMessage = async () => {
    if (!message.trim()) return;
    
    // clear 명령어 감지
    if (message.trim().toLowerCase() === "clear") {
      await handleClearCommand();
      return;
    }
    
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
      
      // LLM 응답 완료 후 문제 목록 새로고침
      setTimeout(() => onMessageSent(), 1000);  // 1초 후 문제 목록 새로고침
    } catch (error) {
      console.error("백엔드 API 호출 실패:", error);
      const errorMessage = { role: "assistant", content: "죄송합니다. 백엔드 서버에 연결할 수 없습니다." };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearCommand = async () => {
    const userMessage = { role: "user", content: "clear" };
    setMessages(prev => [...prev, userMessage]);
    setMessage("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/clear", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage = { 
        role: "assistant", 
        content: data.message || "✅ 세션과 서비스 락이 초기화되었습니다." 
      };
      setMessages(prev => [...prev, assistantMessage]);
      
      // 문제 목록도 새로고침
      setTimeout(() => onMessageSent(), 1000);
    } catch (error) {
      console.error("Clear 명령 실행 실패:", error);
      const errorMessage = { 
        role: "assistant", 
        content: "❌ 초기화에 실패했습니다. 서버 연결을 확인해주세요." 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
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
          
          {/* 메시지 영역 */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500">
                <p>안녕하세요! 질문을 입력해주세요.</p>
                <p className="text-sm mt-2">예: "소프트웨어 설계 3문제 만들어줘"</p>
                <p className="text-xs mt-1 text-gray-400">💡 팁: "clear"를 입력하면 세션과 서비스 락이 초기화됩니다</p>
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
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-semibold"
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
