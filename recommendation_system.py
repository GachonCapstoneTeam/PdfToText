import os
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np
from pathlib import Path
import random
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import json # JSON 모듈 임포트

def clean_change_value(value):
    """
    '상승', '하락' 문자열과 숫자를 +/- 기호와 숫자로 변환하는 함수.
    예: "상승25" -> "+25", "하락10" -> "-10", "보합" -> "0"
    """
    direction = "+" if "상승" in value else "-" if "하락" in value else ""
    number = ''.join(filter(str.isdigit, value))
    return f"{direction}{number}" if number else "0"

def clean_stock_name(name):
    """
    종목명에서 '*'를 제거하고 양쪽 공백을 제거한 후, 내부 다중 공백을 단일 공백으로 변경하는 함수.
    """
    return ' '.join(name.replace('*', '').split())

def crawl_industry_details(industry_url, industry_name):
    """
    주어진 업종 URL에서 종목 상세 정보를 크롤링하는 함수.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(industry_url, headers=headers)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        
        # 네이버 금융 페이지는 'euc-kr' 인코딩을 사용
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-kr')
        table = soup.select_one('div.box_type_l table.type_5 tbody') # 종목 정보가 담긴 테이블 선택
        
        if not table:
            print(f"테이블을 찾을 수 없습니다: {industry_url}")
            return None
        
        stocks_data = []
        
        for tr in table.find_all('tr'): # 테이블의 각 행(row) 순회
            tds = tr.find_all('td') # 각 행의 모든 셀(column) 가져오기
            if len(tds) >= 10: # 데이터가 충분한 셀이 있는지 확인
                # 종목명 클리닝 및 변수명 변경 (stock_name과의 혼동을 피하기 위함)
                stock_name_val = clean_stock_name(tds[0].text) 
                if stock_name_val and not stock_name_val.isspace(): # 종목명이 유효한지 확인
                    stock_data = {
                        '업종명': industry_name,
                        '종목명': stock_name_val,
                        '현재가': tds[1].text.strip(),
                        '전일비': clean_change_value(tds[2].text.strip()), # 전일비 값 클리닝
                        '등락률': tds[3].text.strip(),
                        '거래량': tds[6].text.strip(),
                        '거래대금': tds[7].text.strip(),
                        '전일거래량': tds[8].text.strip()
                    }
                    stocks_data.append(stock_data)
        
        return pd.DataFrame(stocks_data)
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None

def get_industry_list():
    """
    네이버 금융에서 업종 목록과 각 업종별 상세 페이지 URL을 가져오는 함수.
    """
    base_url = "https://finance.naver.com"
    target_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong" # 업종별 시세 페이지
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(target_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-kr')
        table = soup.find('table', {'class': 'type_1'}) # 업종 목록 테이블
        
        if not table:
            return []
        
        industries = []
        for tr in table.find_all('tr'):
            td = tr.find('td')
            if td:
                a_tag = td.find('a') # 업종명과 링크를 가진 <a> 태그
                if a_tag and a_tag.get('href'):
                    industry_name = a_tag.text.strip()
                    href = base_url + a_tag.get('href') # 절대 URL로 변환
                    industries.append((industry_name, href))
        
        return industries
    
    except Exception as e:
        print(f"업종 목록 수집 중 오류 발생: {str(e)}")
        return []

def get_industry_details():
    """
    모든 업종에 대해 상세 종목 정보를 크롤링하여 CSV 파일로 저장하는 함수.
    """
    output_filename = f'stock_data.csv'
    
    industries = get_industry_list() # 업종 목록 가져오기
    
    if not industries:
        print("업종 목록을 가져올 수 없습니다.")
        return
    
    all_data = []
    total_industries = len(industries)
    
    for idx, (industry_name, industry_url) in enumerate(industries, 1):
        print(f"\n[{idx}/{total_industries}] {industry_name} 업종 크롤링 중...")
        time.sleep(random.uniform(0.5, 1.5)) # 네이버 서버 부하 감소를 위한 딜레이
        
        df = crawl_industry_details(industry_url, industry_name) # 해당 업종 상세 정보 크롤링
        if isinstance(df, pd.DataFrame) and not df.empty:
            all_data.append(df)
            print(f"{industry_name} 업종: {len(df)}개 종목 수집 완료")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True) # 모든 업종 데이터를 하나로 합침
        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig') # CSV 파일로 저장
        print(f"\n크롤링 완료! 총 {len(final_df)}개 종목이 {output_filename}에 저장되었습니다.")
        print(f"저장된 파일 경로: {output_filename}")

def load_data(file_path):
    """
    CSV 파일에서 주식 데이터를 로드하고, 특정 컬럼을 숫자형으로 변환하는 함수.
    """
    df = pd.read_csv(file_path)
    
    def convert_to_numeric(value):
        """문자열 값을 숫자로 변환. 변환 실패 시 None 반환."""
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "") # 콤마, % 제거
            try:
                return float(value)
            except ValueError:
                return None # 숫자 변환 불가능 시 None
        return value
    
    # 숫자형으로 변환할 컬럼 목록
    numeric_columns = ["현재가", "전일거래량", "등락률", "거래량", "거래대금"]
    for col in numeric_columns:
        if col in df.columns: # 컬럼 존재 확인
            df[col] = df[col].apply(convert_to_numeric)
    
    # 숫자형으로 변환된 컬럼에서 NA 값(변환 실패 등)이 있는 행 제거
    df.dropna(subset=[col for col in numeric_columns if col in df.columns], inplace=True) 
    return df

def calculate_similarity(df, num_recommendations=10):
    """
    주식 데이터 간의 유사도를 계산하여 추천 종목을 선정하는 함수.
    주요 로직:
    1. 스케일링: '전일거래량', '등락률' 컬럼을 표준화.
    2. 거리 계산: 유클리드 거리를 사용하여 종목 간 거리 행렬 생성.
    3. 추천 로직:
        - 우선 순위 1: 동일 업종 내에서 현재 종목보다 하락률이 더 큰 종목 (하락률 낮은 순).
        - 우선 순위 2: 다른 업종에서 현재 종목보다 하락률이 더 큰 종목 (하락률 낮은 순).
        - 우선 순위 3 (위에서 부족할 시):
            - 동일 업종 내 나머지 종목 (유사도 높은 순).
            - 다른 업종 내 나머지 종목 (유사도 높은 순).
    """
    if len(df) < 2:
        print("유사도 계산을 위한 데이터가 충분하지 않습니다.")
        return {stock_name_iter: [] for stock_name_iter in df["종목명"]}

    scaler = StandardScaler()
    # 스케일링 대상 컬럼 (존재하는 컬럼만 사용)
    scalable_cols = [col for col in ["전일거래량", "등락률"] if col in df.columns]
    if not scalable_cols:
        print("스케일링할 수 있는 컬럼이 없습니다. ('전일거래량', '등락률')")
        return {stock_name_iter: [] for stock_name_iter in df["종목명"]}

    df_scaled = df.copy()
    df_scaled[scalable_cols] = scaler.fit_transform(df_scaled[scalable_cols])
    
    # 스케일링된 데이터를 바탕으로 유클리드 거리 행렬 계산
    distance_matrix = cdist(df_scaled[scalable_cols], df_scaled[scalable_cols], metric="euclidean")
    
    recommendations = {}
    for idx, stock_name_iter in enumerate(df["종목명"]): # 각 종목에 대해 반복
        original_df_idx = df.index[idx] # 원본 DataFrame에서의 현재 종목 인덱스
        industry = df.loc[original_df_idx, "업종명"] 
        current_stock_rate = df.loc[original_df_idx, "등락률"]
        
        # 동일 업종 종목 인덱스 리스트
        same_industry_indices = df[df["업종명"] == industry].index.tolist()
        # 동일 업종 내, 현재 종목보다 하락률이 더 큰(즉, 더 많이 하락한) 종목 (하락률 오름차순 정렬)
        same_industry_higher_decline = [i for i in same_industry_indices 
                                       if i != original_df_idx and df.loc[i, "등락률"] < current_stock_rate]
        same_industry_higher_decline.sort(key=lambda i: df.loc[i, "등락률"])
        
        other_industries_higher_decline = []
        if len(same_industry_higher_decline) < num_recommendations:
            # 다른 업종 종목 중, 현재 종목보다 하락률이 더 큰 종목 (하락률 오름차순 정렬)
            other_industry_indices = [i for i in df.index 
                                     if i != original_df_idx and i not in same_industry_indices 
                                     and df.loc[i, "등락률"] < current_stock_rate]
            other_industry_indices.sort(key=lambda i: df.loc[i, "등락률"])
            other_industries_higher_decline = other_industry_indices
        
        # 하락률 기반 우선 추천 목록 결합
        higher_decline_indices = same_industry_higher_decline + other_industries_higher_decline
        
        # 거리 행렬에서 현재 종목의 행을 참조하기 위한 인덱스
        # df.index는 원본 DataFrame의 인덱스이며, distance_matrix는 0부터 시작하는 정수 인덱스를 사용
        current_stock_dist_matrix_idx = df.index.get_loc(original_df_idx)

        if len(higher_decline_indices) < num_recommendations:
            # 추천 수가 부족할 경우, 유사도 기반으로 추가 추천
            distances = distance_matrix[current_stock_dist_matrix_idx] # 현재 종목과 다른 모든 종목 간의 거리
            
            # 동일 업종 내, 아직 추천되지 않았고, 자기 자신이 아닌 종목 (유사도 높은 순 - 거리 짧은 순)
            remaining_same_industry = [i for i in same_industry_indices 
                                      if i != original_df_idx and i not in higher_decline_indices]
            # distances의 인덱스는 0부터 시작하므로, df.index.get_loc(i)를 사용하여 매핑
            remaining_same_industry.sort(key=lambda i: distances[df.index.get_loc(i)]) 
            
            # 다른 업종 내, 아직 추천되지 않았고, 자기 자신이 아닌 종목 (유사도 높은 순 - 거리 짧은 순)
            remaining_other_industries = [i for i in df.index 
                                         if i != original_df_idx and i not in same_industry_indices 
                                         and i not in higher_decline_indices]
            remaining_other_industries.sort(key=lambda i: distances[df.index.get_loc(i)])
            
            additional_recs_indices = remaining_same_industry + remaining_other_industries
            combined_recs_indices = higher_decline_indices + additional_recs_indices # 최종 추천 후보 인덱스 결합
            
            # 목표 개수만큼 추천 종목명 저장
            recommendations[stock_name_iter] = df["종목명"].loc[combined_recs_indices[:num_recommendations]].tolist()
        else:
            # 하락률 기반 추천만으로도 충분한 경우
            recommendations[stock_name_iter] = df["종목명"].loc[higher_decline_indices[:num_recommendations]].tolist()
            
    return recommendations

def export_recommendations_to_csv(df, recommendations, output_file='stock_recommendations.csv'):
    """
    계산된 추천 결과를 CSV 파일로 저장하는 함수.
    """
    num_recs_per_stock = 0
    if recommendations:
        # 첫 번째 종목의 추천 개수를 기준으로 컬럼 생성 (일반적으로 모든 종목의 추천 개수는 동일)
        num_recs_per_stock = len(next(iter(recommendations.values()))) 
    
    # 추천 종목 컬럼명 생성 (예: 추천종목_1, 추천종목_2, ...)
    rec_columns = [f'추천종목_{i+1}' for i in range(num_recs_per_stock)]
    
    # 결과 DataFrame의 기본 구조 (업종명, 종목명)
    result_df = df[['업종명', '종목명']].copy()
    
    # 추천 종목 컬럼들을 NA 값으로 초기화
    for col in rec_columns:
        result_df[col] = pd.NA # pandas 1.0 이상에서는 pd.NA 사용 권장
    
    # 각 종목별로 추천된 종목들을 해당 행에 채워넣기
    for stock_name_val, recs in recommendations.items(): # 명확성을 위해 변수명 변경 (stock_name -> stock_name_val)
        stock_indices = result_df[result_df['종목명'] == stock_name_val].index
        if not stock_indices.empty:
            idx = stock_indices[0] # 종목명은 고유하다고 가정
            result_df.loc[idx, rec_columns[:len(recs)]] = recs # 실제 추천된 만큼만 채우기
            # 추천 개수가 num_recs_per_stock보다 적은 경우, 나머지 컬럼은 빈 문자열로 처리 (혹은 NA 유지)
            for i in range(len(recs), len(rec_columns)):
                 result_df.loc[idx, rec_columns[i]] = '' # 혹은 pd.NA
        else:
            print(f"경고: 추천 목록을 생성하는 동안 '{stock_name_val}' 종목을 찾을 수 없습니다.")

    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"추천 결과가 {output_file}에 저장되었습니다.")
    return result_df

def recommendation_algorithm():
    """
    전체 추천 알고리즘을 실행하는 메인 함수.
    1. 업종별 주식 데이터 크롤링 (`get_industry_details`).
    2. 크롤링된 데이터 로드 (`load_data`).
    3. 유사도 기반 추천 계산 (`calculate_similarity`).
    4. 추천 결과를 CSV로 저장 (`export_recommendations_to_csv`).
    """
    get_industry_details() # stock_data.csv 생성
    file_path = "stock_data.csv"
    
    if not Path(file_path).exists():
        print(f"{file_path}가 존재하지 않습니다. 크롤링을 먼저 실행해주세요.")
        return

    df = load_data(file_path)
    if df.empty:
        print("데이터를 불러오지 못했거나 데이터가 비어있습니다. 알고리즘을 실행할 수 없습니다.")
        return

    print(f"알고리즘 계산을 시작합니다")
    recommendations = calculate_similarity(df) 
    
    export_recommendations_to_csv(df, recommendations) # stock_recommendations.csv 생성

def crawl_naver_finance_last_search():
    """
    네이버 금융의 '최근 조회 상위' 종목을 크롤링하여 CSV 파일로 저장하는 함수.
    """
    target_url = "https://finance.naver.com/sise/lastsearch2.naver"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(target_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-kr')
        table = soup.select_one('div.box_type_l table.type_5') # 최근 조회 상위 테이블 선택
        
        if not table:
            print("최근 조회 상위 테이블을 찾을 수 없습니다.")
            return None
        
        results = []
        # 테이블의 헤더 행(첫 번째 'tr')을 제외하고 데이터 행만 순회
        for tr in table.find_all('tr')[1:]: 
            tds = tr.find_all('td')
            # 종목명은 보통 두 번째 셀(tds[1])에 있음
            if len(tds) > 1 : 
                name_td = tds[1] 
                if name_td and name_td.find('a'): # <a> 태그 안에 종목명이 있음
                    stock_name_val = clean_stock_name(name_td.find('a').text)
                    if stock_name_val: # 유효한 종목명인 경우에만 추가
                         results.append([stock_name_val])
        
        if not results:
            print("최근 조회된 종목을 찾을 수 없습니다.")
            return None

        df = pd.DataFrame(results, columns=['종목명'])
        
        df.to_csv('last_searched_stocks.csv', index=False, encoding='utf-8-sig')
        print(f"last_searched_stocks.csv에 저장되었습니다.")
        return df
    
    except Exception as e:
        print(f"최근 조회 상위 크롤링 중 오류 발생: {str(e)}")
        return None

def check_file_date_and_execute(file_path, function_to_execute):
    """
    지정된 파일이 오늘 날짜에 생성되었는지 확인하고,
    오늘 생성되지 않았거나 파일이 없으면 주어진 함수를 실행하는 함수.

    Args:
        file_path (str): 확인할 파일 경로.
        function_to_execute (function): 실행할 함수.

    Returns:
        bool: 함수가 실행되었으면 True, 아니면 False.
    """
    try:
        file_creation_time = os.path.getctime(file_path) # 파일 생성 타임스탬프
        creation_date = datetime.datetime.fromtimestamp(file_creation_time).date() # 파일 생성 날짜
        today = datetime.date.today() # 오늘 날짜
        
        if creation_date != today: # 파일 생성일이 오늘이 아니면
            print(f"파일 생성일: {creation_date}, 오늘 날짜: {today}. 파일이 오늘 생성되지 않았으므로 함수를 실행합니다: {function_to_execute.__name__}")
            function_to_execute()
            return True
        else: # 파일이 오늘 이미 생성되었으면
            print(f"파일({file_path})이 오늘 이미 생성되었습니다. ({function_to_execute.__name__} 실행 안함)")
            return False
            
    except FileNotFoundError: # 파일이 존재하지 않으면
        print(f"파일({file_path})을 찾을 수 없습니다. 함수를 실행합니다: {function_to_execute.__name__}")
        function_to_execute()
        return True
    except Exception as e:
        print(f"파일 날짜 확인 중 에러 발생 ({file_path}): {e}")
        return False # 에러 발생 시 실행 안 함으로 간주

def get_unique_recommendations(
    stock_recommendations_file_path: str, 
    last_searched_file_path: str, 
    target_stock_name: str, 
    num_total_recommendations: int = 5,
    limit_from_algo: int = 3,
    limit_from_searched: int = 2
) -> str:
    """
    특정 종목에 대해 중복되지 않는 추천 종목을 JSON 문자열로 반환하는 함수.
    알고리즘 기반 추천 파일(stock_recommendations.csv)에서 최대 `limit_from_algo`개, 
    최근 조회 상위 종목 파일(last_searched_stocks.csv)에서 최대 `limit_from_searched`개를 선택하여
    총 `num_total_recommendations`개를 목표로 한다.

    Args:
        stock_recommendations_file_path (str): 알고리즘 기반 추천 결과가 담긴 CSV 파일 경로.
        last_searched_file_path (str): 최근 조회 상위 종목 목록이 담긴 CSV 파일 경로.
        target_stock_name (str): 추천을 받고자 하는 기준 종목명.
        num_total_recommendations (int): 반환할 총 추천 종목의 최대 개수.
        limit_from_algo (int): 알고리즘 추천 파일에서 가져올 최대 종목 수.
        limit_from_searched (int): 최근 조회 파일에서 가져올 최대 종목 수.

    Returns:
        str: 중복 없는 추천 종목 리스트를 담은 JSON 문자열. 추천 종목이 없으면 빈 리스트의 JSON 문자열.
    """
    algo_recs_pool = [] # 알고리즘 추천 후보군
    searched_recs_pool = [] # 최근 조회 추천 후보군
    
    # 1. stock_recommendations.csv에서 추천 목록 로드
    if Path(stock_recommendations_file_path).exists():
        try:
            df_algo = pd.read_csv(stock_recommendations_file_path, encoding='utf-8-sig')
            if '종목명' in df_algo.columns:
                stock_data_rows = df_algo[df_algo['종목명'] == target_stock_name]
                if not stock_data_rows.empty:
                    row = stock_data_rows.iloc[0] # 해당 종목의 첫 번째 행 사용
                    # '추천종목_'으로 시작하는 모든 컬럼에서 유효한(NA가 아니고 비어있지 않은) 추천 종목 추출
                    rec_cols = [col for col in df_algo.columns if col.startswith('추천종목_')]
                    temp_recs = [str(row[col]) for col in rec_cols if col in row and pd.notna(row[col]) and str(row[col]).strip()]
                    algo_recs_pool.extend(r for r in temp_recs if r != target_stock_name) # 자기 자신은 추천에서 제외
        except Exception as e:
            print(f"{stock_recommendations_file_path} 파일 처리 중 오류: {e}")
    else:
        print(f"경고: {stock_recommendations_file_path} 파일이 존재하지 않습니다.")

    # 2. last_searched_stocks.csv에서 추천 목록 로드
    if Path(last_searched_file_path).exists():
        try:
            df_searched = pd.read_csv(last_searched_file_path, encoding='utf-8-sig')
            # 파일이 비어있지 않고, 첫 번째 컬럼명이 '종목명'인지 확인 (또는 컬럼이 하나인지 확인)
            if not df_searched.empty and df_searched.columns[0] == '종목명': 
                temp_recs = df_searched['종목명'].dropna().astype(str).tolist()
                searched_recs_pool.extend(r for r in temp_recs if r != target_stock_name) # 자기 자신은 추천에서 제외
        except Exception as e:
            print(f"{last_searched_file_path} 파일 처리 중 오류: {e}")
    else:
        print(f"경고: {last_searched_file_path} 파일이 존재하지 않습니다.")

    # 중복 제거 및 순서 섞기
    # 각 소스 내에서 여러 번 나타난 경우 중복이 있을 수 있으므로 set으로 중복 제거 후 list로 변환
    algo_recs_pool = list(set(algo_recs_pool))
    searched_recs_pool = list(set(searched_recs_pool))
    
    random.shuffle(algo_recs_pool) # 추천 순서에 다양성을 주기 위해 섞음
    random.shuffle(searched_recs_pool)

    chosen_recommendations = [] # 최종 선택된 추천 종목 리스트
    
    # 3. algo_recs_pool에서 우선적으로 선택 (지정된 한도 내에서)
    taken_from_algo = 0
    for rec in algo_recs_pool:
        if len(chosen_recommendations) < num_total_recommendations and taken_from_algo < limit_from_algo:
            if rec not in chosen_recommendations: # 최종 리스트에 중복되지 않도록 확인
                chosen_recommendations.append(rec)
                taken_from_algo += 1
        else:
            break # 목표 개수를 채웠거나, 알고리즘 한도를 초과하면 중단
            
    # 4. searched_recs_pool에서 선택 (지정된 한도 내에서)
    taken_from_searched = 0
    for rec in searched_recs_pool:
        if len(chosen_recommendations) < num_total_recommendations and taken_from_searched < limit_from_searched:
            if rec not in chosen_recommendations: # 최종 리스트에 중복되지 않도록 확인
                chosen_recommendations.append(rec)
                taken_from_searched += 1
        else:
            break # 목표 개수를 채웠거나, 최근 조회 한도를 초과하면 중단
            
    # 5. 총 추천 개수를 채우지 못한 경우, 남은 슬롯을 채우기 위해 다시 시도
    #    (우선순위: 알고리즘 풀 -> 최근 조회 풀. 각 풀의 한도는 이미 위에서 적용됨)
    #    이를 통해 한 소스가 다른 소스보다 적은 추천을 가졌을 때, 다른 소스가 더 많이 기여할 수 있게 함.
    
    # algo_recs_pool에서 추가로 채우기 시도 (이미 선택된 것은 제외)
    for rec in algo_recs_pool:
        if len(chosen_recommendations) < num_total_recommendations:
            if rec not in chosen_recommendations:
                chosen_recommendations.append(rec)
        else:
            break
            
    # searched_recs_pool에서 추가로 채우기 시도 (이미 선택된 것은 제외)
    for rec in searched_recs_pool:
        if len(chosen_recommendations) < num_total_recommendations:
            if rec not in chosen_recommendations:
                chosen_recommendations.append(rec)
        else:
            break

    if not chosen_recommendations:
        print(f"'{target_stock_name}'에 대한 유효한 추천 종목이 없습니다.")
        return json.dumps([], ensure_ascii=False) # 빈 리스트 반환
        
    # 최종 선택된 추천 목록을 JSON 문자열로 반환 (한글 인코딩 유지)
    return json.dumps(chosen_recommendations, ensure_ascii=False)


def get_multiple_stocks_recommendations(
    stock_recs_file: str, 
    last_searched_file: str, 
    stock_names: list, 
    num_recommendations: int = 5,
    algo_limit: int = 3,
    searched_limit: int = 2
) -> str:
    """
    여러 종목에 대해 각각 중복 없는 N개의 추천 종목을 JSON 문자열로 반환하는 함수.
    
    Args:
        stock_recs_file (str): 알고리즘 기반 추천 결과가 담긴 CSV 파일 경로.
        last_searched_file (str): 최근 조회 상위 종목 목록이 담긴 CSV 파일 경로.
        stock_names (list): 추천을 받고자 하는 종목명 리스트.
        num_recommendations (int): 각 종목당 반환할 총 추천 종목의 최대 개수.
        algo_limit (int): 알고리즘 추천 파일에서 가져올 최대 종목 수.
        searched_limit (int): 최근 조회 파일에서 가져올 최대 종목 수.
        
    Returns:
        str: 각 종목명을 키로 하고, 해당 종목의 추천 리스트를 값으로 하는 딕셔너리를 
             JSON 문자열로 변환하여 반환.
    """
    all_recommendations_dict = {} # 모든 종목의 추천 결과를 담을 딕셔너리
    
    for stock_name_iter in stock_names: # 주어진 각 종목명에 대해 반복
        # 단일 종목에 대한 추천 함수 호출
        json_string_for_single_stock = get_unique_recommendations(
            stock_recs_file, 
            last_searched_file, 
            stock_name_iter, 
            num_recommendations,
            algo_limit,
            searched_limit
        )
        # JSON 문자열을 파이썬 리스트로 변환하여 딕셔너리에 저장
        all_recommendations_dict[stock_name_iter] = json.loads(json_string_for_single_stock)
        
    # 최종 딕셔너리를 JSON 문자열로 변환하여 반환 (한글 인코딩 유지)
    return json.dumps(all_recommendations_dict, ensure_ascii=False)

def print_unique_recommendations(
    stock_recs_file: str, 
    last_searched_file: str, 
    stock_names: list, 
    num_recommendations: int = 5,
    algo_limit: int = 3,
    searched_limit: int = 2
):
    """
    여러 종목의 중복 없는 추천 결과를 콘솔에 출력하는 함수.
    
    Args:
        stock_recs_file (str): 알고리즘 기반 추천 파일 경로.
        last_searched_file (str): 최근 조회 상위 종목 파일 경로.
        stock_names (list): 추천을 원하는 종목명 리스트.
        num_recommendations (int): 각 종목당 출력할 추천 종목 수.
        algo_limit (int): 알고리즘 추천에서 가져올 최대 개수.
        searched_limit (int): 최근 조회에서 가져올 최대 개수.
    """
    # 여러 종목에 대한 추천 결과 가져오기 (JSON 문자열 형태)
    json_string_recommendations = get_multiple_stocks_recommendations(
        stock_recs_file, 
        last_searched_file, 
        stock_names, 
        num_recommendations,
        algo_limit,
        searched_limit
    )
    
    # JSON 문자열을 파이썬 딕셔너리로 변환
    recommendations_dict = json.loads(json_string_recommendations)
    
    print(f"\n=== 종목별 추천 결과 (각 종목당 최대 {num_recommendations}개) ===")
    print(f"(알고리즘 추천 최대 {algo_limit}개, 최근 조회 추천 최대 {searched_limit}개 우선)")
    for stock_name_val, recs_list in recommendations_dict.items():
        if not recs_list: # 추천 목록이 비어있는 경우
            print(f"\n{stock_name_val}: 추천 종목을 찾을 수 없거나 없습니다.")
        else:
            print(f"\n{stock_name_val}의 추천 종목 ({len(recs_list)}개):")
            for i, rec_item in enumerate(recs_list, 1): # 추천 종목 번호 매겨서 출력
                print(f"{i}. {rec_item}")

# --- 메인 실행 흐름 ---
if __name__ == "__main__":
    # 파일 경로 설정
    stock_data_file = "stock_data.csv" # recommendation_algorithm의 입력 소스
    recommendation_output_file = 'stock_recommendations.csv' # recommendation_algorithm의 결과물
    last_searched_file = "last_searched_stocks.csv" # crawl_naver_finance_last_search의 결과물

    # 1. 업종별 종목 데이터 크롤링 (`stock_data.csv` 생성) 및
    #    이를 기반으로 유사도 추천 알고리즘 실행 (`stock_recommendations.csv` 생성)
    #    `recommendation_output_file`이 오늘 생성되지 않았거나 존재하지 않으면 실행.
    #    `recommendation_algorithm()` 함수 내부에서 `stock_data.csv`를 먼저 생성합니다.
    if not Path(recommendation_output_file).exists() or \
       (datetime.datetime.fromtimestamp(os.path.getctime(recommendation_output_file)).date() != datetime.date.today()):
        print(f"{recommendation_output_file}이 오늘 생성되지 않았거나 존재하지 않아, 추천 알고리즘을 실행합니다.")
        # 이 함수는 stock_data.csv를 생성한 후 stock_recommendations.csv를 생성합니다.
        recommendation_algorithm() 
    else:
        print(f"{recommendation_output_file}이 오늘 이미 생성되었습니다.")


    # 2. 네이버 금융 최근 조회 상위 종목 크롤링 (`last_searched_stocks.csv` 생성)
    #    `last_searched_file`이 오늘 생성되지 않았거나 존재하지 않으면 실행.
    check_file_date_and_execute(last_searched_file, crawl_naver_finance_last_search)

    # 사용할 추천 데이터 파일들
    # 위 로직을 통해 파일들이 최신 상태이거나 생성되었음을 가정합니다.
    
    if not Path(recommendation_output_file).exists():
        print(f"필수 파일 {recommendation_output_file}이 없습니다. 추천을 진행할 수 없습니다.")
    elif not Path(last_searched_file).exists():
        print(f"필수 파일 {last_searched_file}이 없습니다. 추천을 진행할 수 없습니다.")
    else:
        # # 추천받고 싶은 종목 리스트 (예시)
        # target_stocks_for_recommendation = ['삼성전자', 'SK하이닉스', '카카오'] 

        # # 여러 종목에 대한 추천 결과 출력
        # print_unique_recommendations(
        #     recommendation_output_file, 
        #     last_searched_file, 
        #     target_stocks_for_recommendation, 
        #     num_recommendations=5, # 총 5개 목표
        #     algo_limit=3,          # stock_recommendations.csv에서 최대 3개
        #     searched_limit=2       # last_searched_stocks.csv에서 최대 2개
        # )

        print("\n--- 단일 종목 JSON 결과 테스트 ---")
        single_stock_name = '삼성전자' # 여기에 테스트하고 싶은 종목명을 입력하세요.
        json_output_single = get_unique_recommendations(
            recommendation_output_file, 
            last_searched_file, 
            single_stock_name,
            num_total_recommendations=5, # 총 5개 추천 목표
            limit_from_algo=3,           # 알고리즘 추천에서 최대 3개
            limit_from_searched=2        # 최근 검색 추천에서 최대 2개
        )
        print(f"'{single_stock_name}' 추천 결과 (JSON): {json_output_single}")