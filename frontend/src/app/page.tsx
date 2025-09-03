"use client";

import { useCoAgent, useCopilotAction } from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useState } from "react";
import { softwareDesignQuestions, Question } from "./test/data/questions";

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

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");
  const [testSessions, setTestSessions] = useState<TestSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  // 🪁 Frontend Actions: https://docs.copilotkit.ai/guides/frontend-actions
  useCopilotAction({
    name: "setThemeColor",
    description: "Set the theme color of the page.",
    parameters: [{
      name: "themeColor",
      description: "The theme color to set. Make sure to pick nice colors.",
      required: true, 
    }],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main style={{ "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties}>
      <YourMainContent 
        themeColor={themeColor} 
        testSessions={testSessions}
        setTestSessions={setTestSessions}
        currentSessionId={currentSessionId}
        setCurrentSessionId={setCurrentSessionId}
      />
    </main>
  );
}

// State of the agent, make sure this aligns with your agent's state.
type AgentState = {
  // 빈 상태로 유지 (향후 확장 가능)
}

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
  // 🪁 Shared State: https://docs.copilotkit.ai/coagents/shared-state
  const {state, setState} = useCoAgent<AgentState>({
    name: "starterAgent",
    initialState: {},
  })

  // 테스트 세션 생성 함수 (AI 동적 생성)
  const createTestSession = async (type: string, title: string, questionCount: number, difficulty: string = 'intermediate', userLevel: string = 'intermediate') => {
    try {
      const response = await fetch('/api/generate-questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: type, count: questionCount, difficulty, userLevel })
      });
      
      const data = await response.json();
      
      if (data.success) {
        const sessionId = `session_${Date.now()}`;
        const newSession: TestSession = {
          id: sessionId,
          type,
          title,
          questions: data.questions,
          currentQuestion: 0,
          answers: {},
          startTime: new Date(),
          isCompleted: false,
          difficulty,
          userLevel,
          metadata: data.metadata
        };
        
        setTestSessions(prev => [...prev, newSession]);
        setCurrentSessionId(sessionId);
      }
    } catch (error) {
      console.error('Failed to create test session:', error);
      // 폴백: 기존 하드코딩된 문제 사용
      const sessionId = `session_${Date.now()}`;
      const newSession: TestSession = {
        id: sessionId,
        type,
        title,
        questions: softwareDesignQuestions.slice(0, questionCount),
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

  //🪁 Generative UI: https://docs.copilotkit.ai/coagents/generative-ui
  useCopilotAction({
    name: "getWeather",
    description: "Get the weather for a given location.",
    available: "disabled",
    parameters: [
      { name: "location", type: "string", required: true },
    ],
    render: ({ args }) => {
      return <WeatherCard location={args.location} themeColor={themeColor} />
    },
  });

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
                오른쪽 채팅창에서 "소프트웨어 설계 20문제 만들어줘"라고 입력하면<br/>
                문제 테스트를 시작할 수 있습니다!
              </p>
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
              {softwareDesignQuestions.map((question, index) => (
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
              ))}
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
          <p className="text-gray-600 mb-6">이 에이전트는 사용자의 질문을 파악하여 정처기 및  농사관련 질문에 답변을 할수있습니다.</p>
          
          <div className="text-left">
            <p className="text-gray-700 font-medium mb-3">예를 들어 다음과 같이 시도해볼 수 있습니다:</p>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">•</span>
                <span><strong>농사 질문:</strong> "오이에는 어떤 병해충이 있어?"</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">•</span>
                <span><strong>정보처리기사 질문:</strong> "소프트웨어 설계 20문제 만들어줘"</span>
              </li>
            </ul>
          </div>
          
          <p className="text-gray-600 text-sm mt-6">
            질문을 입력하시면 질문에 따라 농사관련 질문이나 정처기 관련 질문을 할수있습니다.
          </p>
        </div>
        
        {/* 하단 입력 영역 */}
        <div className="absolute bottom-0 left-0 right-0 p-6">
          <div className="flex items-center space-x-3">
            <div className="flex-1 relative">
              <input
                type="text"
                placeholder="질문을 입력해 주세요"
                className="w-full p-5 pr-16 bg-gray-200 text-gray-800 rounded-full border-none focus:outline-none focus:ring-2 focus:ring-gray-600 shadow-sm text-lg"
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    const input = e.target as HTMLInputElement;
                    const value = input.value.trim();
                    if (value.includes('소프트웨어 설계') && value.includes('20문제')) {
                      createTestSession('소프트웨어 설계', '소프트웨어 설계 문제 테스트', 20, 'intermediate', 'intermediate');
                      input.value = '';
                    } else if (value.includes('데이터베이스') && value.includes('문제')) {
                      const count = value.match(/(\d+)문제/)?.[1] || '15';
                      createTestSession('데이터베이스', '데이터베이스 문제 테스트', parseInt(count), 'intermediate', 'intermediate');
                      input.value = '';
                    } else if (value.includes('알고리즘') && value.includes('문제')) {
                      const count = value.match(/(\d+)문제/)?.[1] || '10';
                      createTestSession('알고리즘', '알고리즘 문제 테스트', parseInt(count), 'intermediate', 'intermediate');
                      input.value = '';
                    }
                  }
                }}
              />
                                <button 
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2 text-gray-500 hover:text-gray-700 transition-colors"
                    onClick={() => {
                      const input = document.querySelector('input[placeholder="질문을 입력해 주세요"]') as HTMLInputElement;
                      const value = input.value.trim();
                      if (value.includes('소프트웨어 설계') && value.includes('20문제')) {
                        createTestSession('소프트웨어 설계', '소프트웨어 설계 문제 테스트', 20, 'intermediate', 'intermediate');
                        input.value = '';
                      } else if (value.includes('데이터베이스') && value.includes('문제')) {
                        const count = value.match(/(\d+)문제/)?.[1] || '15';
                        createTestSession('데이터베이스', '데이터베이스 문제 테스트', parseInt(count), 'intermediate', 'intermediate');
                        input.value = '';
                      } else if (value.includes('알고리즘') && value.includes('문제')) {
                        const count = value.match(/(\d+)문제/)?.[1] || '10';
                        createTestSession('알고리즘', '알고리즘 문제 테스트', parseInt(count), 'intermediate', 'intermediate');
                        input.value = '';
                      }
                    }}
                  >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
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
