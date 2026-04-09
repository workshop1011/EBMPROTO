import streamlit as st
import pandas as pd

if 'csv_storage' not in st.session_state:
    st.session_state['csv_storage'] = {}
if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None

st.title("📋 저장 목록 현황")

# --- [2번 기능: 저장된 CSV 목록 표 제공] ---
st.header("📊 현재 저장소 파일 리스트")

if st.session_state['csv_storage']:
    summary = []
    for name, data in st.session_state['csv_storage'].items():
        summary.append({
            "파일명": name,
            "행(Rows)": len(data),
            "열(Columns)": len(data.columns),
            "상태": "메모리 상주 중"
        })
    
    summary_df = pd.DataFrame(summary)
    
    # 전체 목록을 표로 출력
    st.table(summary_df)
    
    st.info(f"총 {len(summary)}개의 파일이 웹 브라우저 메모리에 저장되어 있습니다.")
else:
    st.warning("저장된 데이터가 없습니다. 먼저 파일을 업로드해 주세요.")

