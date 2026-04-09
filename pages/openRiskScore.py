import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier

st.title("2. 모델 분석 및 시각화 (Model Analysis)")

# 충돌 해결 1: csv_storage 금고에서 데이터 확인
if 'csv_storage' not in st.session_state or not st.session_state['csv_storage']:
    st.warning("먼저 앞선 페이지에서 분석할 CSV 파일을 업로드해 주세요.")
    st.stop()

# 사용자가 분석할 파일을 직접 선택하도록 변경
selected_file = st.selectbox("분석할 데이터를 선택하세요", list(st.session_state['csv_storage'].keys()))
df = st.session_state['csv_storage'][selected_file]

# 충돌 해결 2: Target 컬럼 하드코딩 제거 및 직접 선택 유도
target_col = st.selectbox("예측할 정답(Target) 컬럼을 선택하세요", df.columns, index=len(df.columns)-1)
st.divider()

if st.button("🚀 베이스라인 vs EBM 모델 학습 시작", type="primary"):
    
    with st.spinner("데이터 전처리 및 모델을 학습 중입니다. 데이터 크기에 따라 수 분이 소요될 수 있습니다..."):
        
        # 1. 간단한 전처리 (수치형 변수만 추출, 선택한 타겟 컬럼 제외)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 만약 타겟이 문자열이면 에러 방지를 위해 미리 차단
        if target_col not in numeric_df.columns:
            st.error(f"선택하신 타겟 컬럼 '{target_col}'이 수치형이 아닙니다. 분류할 수 있는 수치형 타겟을 선택해주세요.")
            st.stop()
            
        # 타겟 컬럼에 결측치가 있는 행 제거
        numeric_df = numeric_df.dropna(subset=[target_col])
        
        # 원본 보호를 위해 copy() 사용
        y = numeric_df[target_col].copy() 
        
        # =====================================================================
        # 🌟 변경된 핵심 부분: 에러를 내고 멈추는 대신, 연속형 데이터를 0과 1로 강제 변환!
        # =====================================================================
        if pd.api.types.is_float_dtype(y) or y.nunique() > 20:
            median_val = y.median()
            st.warning(f"⚠️ '{target_col}'은(는) 연속형 데이터입니다. 분류 모델을 돌리기 위해 중앙값({median_val:.3f})을 기준으로 강제로 0(정상)과 1(위험)로 변환합니다.")
            
            # 신용점수 등은 낮을수록 위험하므로: 중앙값 미만 = 1(위험), 이상 = 0(정상)
            y = (y < median_val).astype(int)
            
        # 변환이 끝난 후 고유값 개수 다시 측정 (무조건 2개가 됨)
        y_unique = y.nunique()
        is_multiclass = y_unique > 2
        # =====================================================================
        
        X = numeric_df.drop(columns=[target_col, 'SK_ID_CURR'], errors='ignore')
        
        try:
            # y가 0과 1로 예쁘게 나뉘었기 때문에 stratify(비율 유지) 분할도 완벽하게 작동합니다!
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     

        # ----------------------------------------------------
        # 모델 A: OpenRiskScore 베이스라인 (Logistic Regression)
        # ----------------------------------------------------
        base_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])
        base_pipeline.fit(X_train, y_train)
        
        if is_multiclass:
            base_pred = base_pipeline.predict_proba(X_test)
            try:
                base_auc = roc_auc_score(y_test, base_pred, multi_class='ovr')
            except ValueError:
                base_auc = 0.0 # 극단적 클래스 불균형 등으로 인한 에러 방지
            coefs = base_pipeline.named_steps['model'].coef_
            base_importances = np.mean(np.abs(coefs), axis=0)
        else:
            base_pred = base_pipeline.predict_proba(X_test)[:, 1]
            base_auc = roc_auc_score(y_test, base_pred)
            base_importances = np.abs(base_pipeline.named_steps['model'].coef_[0])

        base_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': base_importances})
        base_imp_df = base_imp_df.sort_values(by='Importance', ascending=False).head(10)

        # ----------------------------------------------------
        # 모델 B: EBM (Explainable Boosting Machine)
        # ----------------------------------------------------
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        # 안정성을 위해 n_jobs=1 (단일 코어) 사용
        ebm = ExplainableBoostingClassifier(random_state=42, n_jobs=1)
        ebm.fit(X_train_imp, y_train)
    
        if is_multiclass:
            ebm_pred = ebm.predict_proba(X_test_imp)
            try:
                ebm_auc = roc_auc_score(y_test, ebm_pred, multi_class='ovr')
            except ValueError:
                ebm_auc = 0.0
        else:
            ebm_pred = ebm.predict_proba(X_test_imp)[:, 1]
            ebm_auc = roc_auc_score(y_test, ebm_pred)

        ebm_global = ebm.explain_global()
        ebm_imp_df = pd.DataFrame({
            'Feature': ebm_global.data()['names'],  # <--- X.columns 대신 이렇게 변경하세요!
            'Importance': ebm_global.data()['scores']
        }).sort_values(by='Importance', ascending=False).head(10)

        # ----------------------------------------------------
        # 4번(LLM 리포트) 페이지로 넘겨줄 데이터 세션 저장
        # ----------------------------------------------------
        st.session_state['model_results'] = {
            'base_auc': base_auc,
            'ebm_auc': ebm_auc,
            'base_top_features': base_imp_df['Feature'].tolist(),
            'ebm_top_features': ebm_imp_df['Feature'].tolist()
        }

    # --- 시각화 레이아웃 ---
    st.success("학습 완료! (결과 데이터가 LLM 리포트 페이지로 전달되었습니다)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("📈 Base Model (Logistic)")
        st.metric("AUC Score", f"{base_auc:.4f}")
        fig1 = go.Figure(go.Bar(x=base_imp_df['Importance'], y=base_imp_df['Feature'], orientation='h'))
        fig1.update_layout(title="Top 10 변수 중요도 (Base)", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.header("🧠 EBM Model")
        st.metric("AUC Score", f"{ebm_auc:.4f}")
        fig2 = go.Figure(go.Bar(x=ebm_imp_df['Importance'], y=ebm_imp_df['Feature'], orientation='h', marker_color='orange'))
        fig2.update_layout(title="Top 10 변수 중요도 (EBM)", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)