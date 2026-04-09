import streamlit as st
import pandas as pd

if 'csv_storage' not in st.session_state:
    st.session_state['csv_storage'] = {}
if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None

st.title("📥 파일 업로드 및 다운로드")

# --- [1번 기능: 업로드 및 임시 저장] ---
st.header("새로운 파일 업로드")
uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type=["csv"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    
    # 🌟 어떤 인코딩이든 다 열어버리는 만능 시도 로직
    # (표준 utf-8 -> 한국어 cp949 -> 서구권 cp1252 -> 최후의 보루 latin1 순서)
    encodings_to_try = ['utf-8', 'cp949', 'cp1252', 'latin1']
    df = None
    
    for enc in encodings_to_try:
        try:
            # 파일을 처음부터 다시 읽기 위해 책갈피 초기화
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            # 성공하면 for문을 빠져나감
            break
        except UnicodeDecodeError:
            # 에러가 나면 다음 인코딩으로 넘어감
            continue
            
    if df is not None:
        # 세션 저장소에 저장
        st.session_state['csv_storage'][file_name] = df
        st.session_state['last_file'] = file_name
        st.success(f"✅ '{file_name}' 파일이 저장소에 등록되었습니다. (적용된 인코딩: {enc})")
    else:
        st.error("알 수 없는 인코딩 형식입니다. 파일을 열 수 없습니다.")

st.divider()

# --- [3번 기능: 저장된 CSV 불러오기 및 다운로드] ---
st.header("저장된 파일 불러오기 / 다운로드")
file_list = list(st.session_state['csv_storage'].keys())

if file_list:
    selected_file = st.selectbox("불러올 파일을 선택하세요", file_list)
    
    if selected_file:
        df_to_show = st.session_state['csv_storage'][selected_file]
        st.write(f"📂 **{selected_file}** 데이터 미리보기")
        st.dataframe(df_to_show, use_container_width=True)
        
        # 다운로드 버튼 구현
        csv_data = df_to_show.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 현재 선택된 파일 다운로드",
            data=csv_data,
            file_name=f"download_{selected_file}",
            mime="text/csv"
        )
else:
    st.info("저장소에 저장된 파일이 없습니다.")