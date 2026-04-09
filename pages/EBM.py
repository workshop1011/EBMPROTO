import streamlit as st
import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

if 'csv_storage' not in st.session_state:
    st.session_state['csv_storage'] = {}

if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None

st.title("🧠 EBM (Explainable Boosting Machine) 분석 및 시각화")
st.write("저장된 CSV 데이터로 EBM 모델을 학습시키고, 결과를 투명하게 해석(Visualize)합니다.")

if 'csv_storage' not in st.session_state or not st.session_state['csv_storage']:
    st.warning("저장소에 데이터가 없습니다. 먼저 파일을 업로드해 주세요.")
    st.stop()

selected_file = st.selectbox("분석할 데이터를 선택하세요", list(st.session_state['csv_storage'].keys()))
df = st.session_state['csv_storage'][selected_file]

target_col = st.selectbox("예측할 정답(Target) 컬럼을 선택하세요", df.columns, index=len(df.columns)-1)
st.divider()

# ==========================================
# 1. 모델 학습 섹션
# ==========================================
if st.button("🚀 모델 학습 시작", type="primary"):
    target_dtype = df[target_col].dtype
    unique_count = df[target_col].nunique()
    
    if unique_count <= 1:
        st.error(f"❌ 타겟 컬럼('{target_col}')의 고유값이 1개 이하입니다. 학습할 수 없습니다.")
        st.stop()

    if pd.api.types.is_float_dtype(target_dtype) or unique_count > 20:
        task_type = "Regression"
    else:
        task_type = "Classification"

    try:
        df_clean = df.fillna(0)
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        with st.spinner("⏳ EBM 모델을 학습하는 중입니다. (안정성을 위해 단일 코어로 작동합니다)..."):
            
            # 🌟 에러 수정: n_jobs=1 로 설정하여 윈도우 메모리 충돌 방지
            if task_type == "Classification":
                ebm = ExplainableBoostingClassifier(random_state=42, n_jobs=1)
                ebm.fit(X_train, y_train)
                proba = ebm.predict_proba(X_test)
                score = roc_auc_score(y_test, proba[:, 1]) if proba.shape[1] == 2 else roc_auc_score(y_test, proba, multi_class='ovr')
                metric_label = "AUC 점수"
            else:
                ebm = ExplainableBoostingRegressor(random_state=42, n_jobs=1)
                ebm.fit(X_train, y_train)
                preds = ebm.predict(X_test)
                score = r2_score(y_test, preds) 
                metric_label = "R² Score"
                
        st.success("🎉 학습 완료!")
        st.metric(label=f"테스트 데이터 성능 ({metric_label})", value=f"{score:.3f}")
        
        # 🌟 핵심: 화면이 새로고침되어도 차트가 날아가지 않도록 세션에 저장
        st.session_state['trained_ebm'] = ebm
        st.session_state['ebm_global'] = ebm.explain_global()  # 전체 설명서 생성
        st.session_state['feature_names'] = X.columns.tolist()

    except Exception as e:
        st.error(f"🚨 예상치 못한 시스템 오류가 발생했습니다. (에러 코드: {type(e).__name__})")
        with st.expander("오류 상세 내용 (Traceback 보기)"):
            st.code(traceback.format_exc(), language="python")

# ==========================================
# 2. 결과 시각화 섹션 (학습된 모델이 있을 때만 표시)
# ==========================================
if 'trained_ebm' in st.session_state:
    st.divider()
    st.header("📊 EBM 모델 해석 (Explainability)")
    
    ebm_global = st.session_state['ebm_global']
    features = st.session_state['feature_names']

    # (1) 전체 변수 중요도 요약 차트
    st.subheader("1. 전체 변수 중요도 (Overall Feature Importance)")
    st.write("결과 예측에 가장 큰 영향을 미친 핵심 변수들의 순위입니다.")
    fig_importance = ebm_global.visualize()  # 인자를 넣지 않으면 전체 중요도 반환
    st.plotly_chart(fig_importance, use_container_width=True)

    st.divider()

    # (2) 개별 변수 상세 영향력 차트 (Shape Function)
    st.subheader("2. 개별 변수의 영향력 (Shape Function)")
    st.write("특정 변수의 값이 변할 때, 예측 결과에 어떤 플러스/마이너스 영향을 주는지 보여줍니다.")
    
    # 콤보박스로 확인하고 싶은 변수 선택
    selected_feature = st.selectbox("상세히 분석할 변수를 선택하세요", features)
    feature_index = features.index(selected_feature)
    
    # 선택된 변수의 차트 추출 및 표시
    fig_shape = ebm_global.visualize(feature_index)
    st.plotly_chart(fig_shape, use_container_width=True)