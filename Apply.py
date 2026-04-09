import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
import time
from datetime import date
from io import BytesIO



# [위치 선정] - 페이지 링크/네비게이션 설정 영역

# 🌟 반드시 첫 번째 명령어로 실행되어야 함
st.set_page_config(page_title="CSV 마스터 시스템", layout="wide")

# 세션 상태 초기화 (모든 페이지에서 공유됨)
if 'csv_storage' not in st.session_state:
    st.session_state['csv_storage'] = {}
if 'last_file' not in st.session_state:
    st.session_state['last_file'] = None

st.title("🚀 CSV 통합 관리 시스템")
st.write("왼쪽 사이드바 메뉴를 통해 기능을 선택해 주세요.")

# 메인 화면에는 가장 최근 파일을 가볍게 표시 (요청 0번 기능)
st.divider()
st.header("🕒 최근 업로드 데이터 요약")

if st.session_state['last_file']:
    last_fn = st.session_state['last_file']
    st.info(f"가장 최근에 작업한 파일: **{last_fn}**")
    st.dataframe(st.session_state['csv_storage'][last_fn].head(), use_container_width=True)
else:
    st.warning("아직 업로드된 파일이 없습니다. 'download' 페이지에서 파일을 먼저 올려주세요.")