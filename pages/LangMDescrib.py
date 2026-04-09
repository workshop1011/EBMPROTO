import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate  # <--- 중간에 '_core'를 추가!

st.title("3. LLM 기반 해석 리포트 (LangChain Report)")

# ==========================================
# 🌟 변경점: 사이드바로 API 키 입력란 완전 분리
# ==========================================
with st.sidebar:
    st.header("⚙️ 환경 설정 (Settings)")
    st.write("리포트 생성을 위해 API 키를 등록해 주세요.")
    
    # 세션에 키가 저장되어 있다면 기본값으로 불러오기
    saved_key = st.session_state.get('openai_api_key', '')
    
    # 텍스트 인풋을 사이드바에 배치
    api_key_input = st.text_input(
        label="OpenAI API Key 입력", 
        type="password", 
        value=saved_key,
        placeholder="sk-..."
    )
    
    # 입력된 키를 세션에 저장 (페이지를 이동해도 유지됨)
    if api_key_input:
        st.session_state['openai_api_key'] = api_key_input
        st.success("✅ 키가 등록되었습니다.")
    else:
        st.warning("⚠️ API 키가 필요합니다.")

# ==========================================
# 메인 화면 로직
# ==========================================
st.write("앞서 분석한 두 모델의 결과를 바탕으로 AI가 종합 해석 리포트를 작성합니다.")

# 2번 페이지에서 넘어온 데이터가 있는지 확인
if not st.session_state.get('model_results'):
    st.warning("먼저 2번 페이지(Model Analysis)에서 모델 학습을 진행해 주세요.")
    st.stop()

# API 키가 없으면 메인 화면에 안내 메시지 띄우고 버튼 숨기기
if not st.session_state.get('openai_api_key'):
    st.info("👈 좌측 사이드바에서 OpenAI API Key를 먼저 입력해 주세요.")
    st.stop()

# API 키도 있고 데이터도 있을 때만 버튼 활성화
if st.button("📝 자동 해석 리포트 생성", type="primary"):
    results = st.session_state['model_results']
    current_api_key = st.session_state['openai_api_key']
    
    with st.spinner("AI가 결과를 분석하고 리포트를 작성 중입니다. (약 10~20초 소요)..."):
        try:
            # LangChain 모델 초기화
            llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o-mini", api_key=current_api_key)
            
            # 프롬프트 엔지니어링
            prompt = PromptTemplate(
                input_variables=["base_auc", "ebm_auc", "base_features", "ebm_features"],
                template="""
                당신은 금융권 최고 수준의 데이터 분석가 및 리스크 평가 전문가입니다.
                다음은 신용 평가(Default Risk) 데이터를 사용하여 두 가지 모델을 학습시킨 결과입니다.

                [분석 결과]
                1. 베이스라인 모델 (Logistic Regression)
                   - 성능(AUC): {base_auc}
                   - 판단에 가장 큰 영향을 미친 상위 변수들: {base_features}
                
                2. 설명 가능한 AI 모델 (Explainable Boosting Machine)
                   - 성능(AUC): {ebm_auc}
                   - 판단에 가장 큰 영향을 미친 상위 변수들: {ebm_features}
                   
                이 데이터를 바탕으로 다음 내용을 포함하는 전문가 수준의 분석 리포트를 한국어로 작성해 주세요:
                1. 두 모델의 성능(AUC) 비교에 대한 평가
                2. 두 모델이 중요하게 생각한 변수(Feature)들의 차이점 분석
                3. 실제 금융 실무진(심사역)에게 EBM 모델의 도입을 설득할 때 강조할 수 있는 장점
                
                마크다운 형식으로 가독성 좋게 작성해 주세요.
                """
            )
            
            # 파이프라인 실행
            chain = prompt | llm
            response = chain.invoke({
                "base_auc": round(results['base_auc'], 4),
                "ebm_auc": round(results['ebm_auc'], 4),
                "base_features": ", ".join(results['base_top_features'][:5]),
                "ebm_features": ", ".join(results['ebm_top_features'][:5])
            })
            
            st.divider()
            st.subheader("📊 AI 모델 평가 종합 리포트")
            st.markdown(response.content)
            
        except Exception as e:
            st.error(f"🚨 API 호출 중 오류가 발생했습니다. 사이드바의 API 키가 정확한지 확인해 주세요.\n\n오류 상세 내용: {e}")