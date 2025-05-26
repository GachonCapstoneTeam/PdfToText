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
    direction = "+" if "상승" in value else "-" if "하락" in value else ""
    number = ''.join(filter(str.isdigit, value))
    return f"{direction}{number}" if number else "0"

def clean_stock_name(name):
    return ' '.join(name.replace('*', '').split())

def crawl_industry_details(industry_url, industry_name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(industry_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-kr')
        table = soup.select_one('div.box_type_l table.type_5 tbody')
        
        if not table:
            print(f"테이블을 찾을 수 없습니다: {industry_url}")
            return None
        
        stocks_data = []
        
        for tr in table.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) >= 10:
                stock_name = clean_stock_name(tds[0].text)
                if stock_name and not stock_name.isspace():
                    stock_data = {
                        '업종명': industry_name,
                        '종목명': stock_name,
                        '현재가': tds[1].text.strip(),
                        '전일비': clean_change_value(tds[2].text.strip()),
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
    base_url = "https://finance.naver.com"
    target_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(target_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-kr')
        table = soup.find('table', {'class': 'type_1'})
        
        if not table:
            return []
        
        industries = []
        for tr in table.find_all('tr'):
            td = tr.find('td')
            if td:
                a_tag = td.find('a')
                if a_tag and a_tag.get('href'):
                    industry_name = a_tag.text.strip()
                    href = base_url + a_tag.get('href')
                    industries.append((industry_name, href))
        
        return industries
    
    except Exception as e:
        print(f"업종 목록 수집 중 오류 발생: {str(e)}")
        return []

def get_industry_details():
    output_filename = f'stock_data.csv'
    
    industries = get_industry_list()
    
    if not industries:
        print("업종 목록을 가져올 수 없습니다.")
        return
    
    all_data = []
    total_industries = len(industries)
    
    for idx, (industry_name, industry_url) in enumerate(industries, 1):
        print(f"\n[{idx}/{total_industries}] {industry_name} 업종 크롤링 중...")
        time.sleep(1) # 네이버 서버 부하 감소를 위한 딜레이
        
        df = crawl_industry_details(industry_url, industry_name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            all_data.append(df)
            print(f"{industry_name} 업종: {len(df)}개 종목 수집 완료")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n크롤링 완료! 총 {len(final_df)}개 종목이 {output_filename}에 저장되었습니다.")
        print(f"저장된 파일 경로: {output_filename}")

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    def convert_to_numeric(value):
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "")
            try:
                return float(value)
            except ValueError:
                return None
        return value
    
    numeric_columns = ["현재가", "전일거래량", "등락률"] # '거래량', '거래대금' 등도 필요시 추가
    for col in numeric_columns:
        df[col] = df[col].apply(convert_to_numeric)
    
    df.dropna(subset=numeric_columns, inplace=True) # 숫자 변환 실패한 행 제거
    return df

def calculate_similarity(df, num_recommendations=10):
    # 데이터가 충분한지 확인
    if len(df) < 2: # 최소 2개 종목은 있어야 거리 계산 가능
        print("유사도 계산을 위한 데이터가 충분하지 않습니다.")
        return {stock: [] for stock in df["종목명"]}


    scaler = StandardScaler()
    # 스케일링할 컬럼이 실제 df에 있는지 확인
    scalable_cols = [col for col in ["전일거래량", "등락률"] if col in df.columns]
    if not scalable_cols:
        print("스케일링할 수 있는 컬럼이 없습니다. ('전일거래량', '등락률')")
        # 모든 종목에 대해 빈 리스트 반환
        return {stock: [] for stock in df["종목명"]}

    df_scaled = df.copy()
    df_scaled[scalable_cols] = scaler.fit_transform(df_scaled[scalable_cols])
    
    distance_matrix = cdist(df_scaled[scalable_cols], df_scaled[scalable_cols], metric="euclidean")
    
    recommendations = {}
    for idx, stock_name_iter in enumerate(df["종목명"]): # 변수명 변경 (stock -> stock_name_iter)
        industry = df.loc[df.index[idx], "업종명"] # df.index[idx] 사용
        current_stock_rate = df.loc[df.index[idx], "등락률"]
        
        same_industry_indices = df[df["업종명"] == industry].index.tolist()
        same_industry_higher_decline = [i for i in same_industry_indices 
                                       if i != df.index[idx] and df.loc[i, "등락률"] < current_stock_rate]
        same_industry_higher_decline.sort(key=lambda i: df.loc[i, "등락률"])
        
        other_industries_higher_decline = []
        if len(same_industry_higher_decline) < num_recommendations:
            other_industry_indices = [i for i in df.index 
                                     if i != df.index[idx] and i not in same_industry_indices 
                                     and df.loc[i, "등락률"] < current_stock_rate]
            other_industry_indices.sort(key=lambda i: df.loc[i, "등락률"])
            other_industries_higher_decline = other_industry_indices
        
        higher_decline_indices = same_industry_higher_decline + other_industries_higher_decline
        
        if len(higher_decline_indices) < num_recommendations:
            distances = distance_matrix[idx]
            
            remaining_same_industry = [i for i in same_industry_indices 
                                      if i != df.index[idx] and i not in higher_decline_indices]
            remaining_same_industry.sort(key=lambda i: distances[df.index.get_loc(i)]) # distances는 원래 인덱스 기준
            
            remaining_other_industries = [i for i in df.index 
                                         if i != df.index[idx] and i not in same_industry_indices 
                                         and i not in higher_decline_indices]
            remaining_other_industries.sort(key=lambda i: distances[df.index.get_loc(i)]) # distances는 원래 인덱스 기준
            
            additional_recs_indices = remaining_same_industry + remaining_other_industries
            combined_recs_indices = higher_decline_indices + additional_recs_indices
            
            # df.index를 사용하여 종목명 가져오기
            recommendations[stock_name_iter] = df["종목명"].loc[combined_recs_indices[:num_recommendations]].tolist()
        else:
            recommendations[stock_name_iter] = df["종목명"].loc[higher_decline_indices[:num_recommendations]].tolist()
            
    return recommendations

def export_recommendations_to_csv(df, recommendations, output_file='stock_recommendations.csv'):
    rec_columns = [f'추천종목_{i+1}' for i in range(10)] # num_recommendations에 맞게 조절 가능
    
    result_df = df[['업종명', '종목명']].copy()
    
    for col in rec_columns:
        result_df[col] = pd.NA # 빈 값을 NA로 초기화 (추후 ''로 채워짐)
    
    for stock, recs in recommendations.items():
        # df에서 stock의 인덱스를 찾아 result_df에 동일하게 적용
        stock_indices = result_df[result_df['종목명'] == stock].index
        if not stock_indices.empty:
            idx = stock_indices[0]
            # 추천 종목 수가 rec_columns 수보다 적을 수 있으므로 슬라이싱으로 채움
            result_df.loc[idx, rec_columns[:len(recs)]] = recs
            # 남는 추천종목 컬럼은 빈 문자열로 채우거나 pd.NA 유지
            for i in range(len(recs), len(rec_columns)):
                 result_df.loc[idx, rec_columns[i]] = '' # 또는 pd.NA
        else:
            print(f"경고: 추천 목록을 생성하는 동안 '{stock}' 종목을 찾을 수 없습니다.")

    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"추천 결과가 {output_file}에 저장되었습니다.")
    return result_df

def recommendation_algorithm():
    get_industry_details() # 데이터 크롤링 및 stock_data.csv 생성
    file_path = "stock_data.csv"
    
    if not Path(file_path).exists():
        print(f"{file_path}가 존재하지 않습니다. 크롤링을 먼저 실행해주세요.")
        return

    df = load_data(file_path)
    if df.empty:
        print("데이터를 불러오지 못했거나 데이터가 비어있습니다. 알고리즘을 실행할 수 없습니다.")
        return

    print(f"알고리즘 계산을 시작합니다")
    recommendations = calculate_similarity(df) # df 그대로 전달
    
    export_recommendations_to_csv(df, recommendations) # df 그대로 전달

def crawl_naver_finance_last_search():
    target_url = "https://finance.naver.com/sise/lastsearch2.naver"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(target_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='euc-kr')
        table = soup.select_one('div.box_type_l table.type_5')
        
        if not table:
            print("최근 조회 상위 테이블을 찾을 수 없습니다.") # print문으로 변경
            return None # 오류 시 None 반환
        
        results = []
        for tr in table.find_all('tr')[1:]: # 헤더 제외
            tds = tr.find_all('td')
            if len(tds) > 1 : # 종목명 td가 있는지 확인 (보통 2번째 td)
                name_td = tds[1] # 종목명은 보통 2번째 td에 위치
                if name_td and name_td.find('a'): # a 태그가 있는 경우만
                    stock_name = clean_stock_name(name_td.find('a').text) # clean_stock_name 적용
                    if stock_name: # 빈 문자열이 아닌 경우만 추가
                         results.append([stock_name])
        
        if not results:
            print("최근 조회된 종목을 찾을 수 없습니다.")
            return None

        df = pd.DataFrame(results, columns=['종목명']) # 컬럼명 지정
        
        df.to_csv('last_searched_stocks.csv', index=False, encoding='utf-8-sig') # header=True 명시적 표현
        print(f"last_searched_stocks.csv에 저장되었습니다.")
        return df
    
    except Exception as e:
        print(f"최근 조회 상위 크롤링 중 오류 발생: {str(e)}")
        return None

def check_file_date_and_execute(file_path, function_to_execute):
    try:
        file_creation_time = os.path.getctime(file_path)
        creation_date = datetime.datetime.fromtimestamp(file_creation_time).date()
        today = datetime.date.today()
        
        if creation_date != today:
            print(f"파일 생성일: {creation_date}, 오늘 날짜: {today}. 파일이 오늘 생성되지 않았으므로 함수를 실행합니다: {function_to_execute.__name__}")
            function_to_execute()
            return True
        else:
            print(f"파일({file_path})이 오늘 이미 생성되었습니다. ({function_to_execute.__name__} 실행 안함)")
            return False
            
    except FileNotFoundError:
        print(f"파일({file_path})을 찾을 수 없습니다. 함수를 실행합니다: {function_to_execute.__name__}")
        function_to_execute()
        return True
    except Exception as e:
        print(f"파일 날짜 확인 중 에러 발생 ({file_path}): {e}")
        return False

def get_unique_recommendations(file_paths, stock_name, num_recommendations=5):
    """
    특정 종목에 대해 중복되지 않는 추천 종목을 JSON 문자열로 반환하는 함수.
    여러 CSV 파일에서 데이터를 읽어옵니다.

    Args:
        file_paths (list): 추천 종목 CSV 파일 경로 리스트
        stock_name (str): 추천을 원하는 종목명
        num_recommendations (int): 반환할 추천 종목 수 (기본값: 5)

    Returns:
        str: 중복 없는 추천 종목 리스트의 JSON 문자열. 
             종목이 존재하지 않거나 추천이 없으면 빈 리스트의 JSON 문자열 "[]" 반환.
    """
    try:
        collected_recommendations = []
        found_stock_in_any_file = False
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                print(f"경고: {file_path} 파일이 존재하지 않습니다. 건너뜁니다.")
                continue
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')

                if '종목명' in df.columns and any(col.startswith('추천종목_') for col in df.columns):
                    # stock_recommendations.csv와 유사한 구조
                    stock_data_rows = df[df['종목명'] == stock_name]
                    if not stock_data_rows.empty:
                        found_stock_in_any_file = True
                        row = stock_data_rows.iloc[0]
                        rec_columns = [f'추천종목_{i+1}' for i in range(10)] # 최대 10개로 가정
                        file_specific_recs = [str(row[col]) for col in rec_columns if col in row and pd.notna(row[col]) and str(row[col]).strip()]
                        collected_recommendations.extend(file_specific_recs)

                elif len(df.columns) == 1 and '종목명' in df.columns or df.shape[1] == 1:
                    # last_searched_stocks.csv와 유사한 구조 (단일 컬럼, 종목명 리스트)
                    # 첫 번째 컬럼을 종목명으로 간주
                    stock_list = df.iloc[:, 0].dropna().astype(str).tolist()
                    # 이 경우, 파일 자체가 추천 목록으로 간주될 수 있음
                    # 만약 stock_name이 이 리스트에 있다면, 다른 종목들을 추천으로 간주할 수 있음
                    if stock_name in stock_list:
                        found_stock_in_any_file = True # 해당 파일에 관심 종목이 있음을 표시
                        # stock_name 자신을 제외한 나머지 종목들을 추천 후보로 추가
                        collected_recommendations.extend([s for s in stock_list if s != stock_name])
                    
            except Exception as e:
                print(f"{file_path} 파일 처리 중 오류 발생: {e}")

        if not found_stock_in_any_file and not collected_recommendations: # 어떤 파일에서도 종목을 못 찾았고, 수집된 추천도 없을 때
            print(f"'{stock_name}' 종목을 어떤 파일에서도 찾을 수 없거나 관련 추천 데이터가 없습니다.")
            return json.dumps([], ensure_ascii=False)

        if not collected_recommendations:
            print(f"'{stock_name}' 종목에 대한 추천 종목이 없습니다.")
            return json.dumps([], ensure_ascii=False)

        # 중복 제거 및 자기 자신 제외
        unique_recs = list(set(s for s in collected_recommendations if s != stock_name and s)) # 빈 문자열도 제외

        if not unique_recs:
            print(f"'{stock_name}'에 대한 유효한 추천 종목이 없습니다 (자기 자신 제외, 중복 제거 후).")
            return json.dumps([], ensure_ascii=False)
            
        final_recommendations = random.sample(unique_recs, min(len(unique_recs), num_recommendations))
        return json.dumps(final_recommendations, ensure_ascii=False)

    except Exception as e:
        print(f"'{stock_name}'에 대한 추천 종목 조회 중 전역 오류 발생: {e}")
        return json.dumps([], ensure_ascii=False)


def get_multiple_stocks_recommendations(file_paths, stock_names, num_recommendations=5):
    """
    여러 종목에 대해 각각 중복 없는 N개의 추천 종목을 JSON 문자열로 반환하는 함수.
    
    Args:
        file_paths (list): 추천 종목 CSV 파일 경로 리스트
        stock_names (list): 추천을 원하는 종목명 리스트
        num_recommendations (int): 각 종목당 반환할 추천 종목 수 (기본값: 5)
        
    Returns:
        str: 종목별 중복 없는 추천 종목 딕셔너리를 나타내는 JSON 문자열.
             {'종목A': ['추천1', '추천2'], '종목B': ['추천3']} 형식.
    """
    all_recommendations_dict = {}
    
    for stock_name_iter in stock_names: # 변수명 변경 (stock -> stock_name_iter)
        # get_unique_recommendations는 JSON 문자열(배열 형태)을 반환
        json_string_for_single_stock = get_unique_recommendations(file_paths, stock_name_iter, num_recommendations)
        # JSON 문자열을 파이썬 리스트로 변환하여 딕셔너리에 저장
        all_recommendations_dict[stock_name_iter] = json.loads(json_string_for_single_stock)
        
    # 최종 딕셔너리를 JSON 문자열로 변환하여 반환
    return json.dumps(all_recommendations_dict, ensure_ascii=False)

def print_unique_recommendations(file_paths, stock_names, num_recommendations=5):
    """
    여러 종목의 중복 없는 추천 결과를 출력하는 함수.
    get_multiple_stocks_recommendations로부터 JSON 문자열을 받아 처리합니다.
    
    Args:
        file_paths (list): 추천 종목 CSV 파일 경로 리스트
        stock_names (list): 추천을 원하는 종목명 리스트
        num_recommendations (int): 각 종목당 출력할 추천 종목 수 (기본값: 5)
    """
    # get_multiple_stocks_recommendations는 JSON 문자열을 반환
    json_string_recommendations = get_multiple_stocks_recommendations(file_paths, stock_names, num_recommendations)
    
    # JSON 문자열을 파이썬 딕셔너리로 파싱
    recommendations_dict = json.loads(json_string_recommendations)
    
    print(f"\n=== 종목별 추천 결과 (각 종목당 최대 {num_recommendations}개, 중복 없음) ===")
    for stock, recs_list in recommendations_dict.items(): # recs -> recs_list로 명확화
        if not recs_list:  # 추천 종목 리스트가 비어있는 경우
            print(f"\n{stock}: 추천 종목을 찾을 수 없거나 없습니다.")
        else:
            print(f"\n{stock}의 추천 종목:")
            for i, rec_item in enumerate(recs_list, 1): # rec -> rec_item으로 명확화
                print(f"{i}. {rec_item}")

# --- Main execution flow ---
if __name__ == "__main__":
    # 파일 경로 설정
    stock_data_file = "stock_data.csv"
    last_searched_file = "last_searched_stocks.csv"
    recommendation_output_file = 'stock_recommendations.csv'

    # 1. 업종별 종목 데이터 크롤링 및 stock_data.csv 생성 (오늘 날짜 파일 없으면 실행)
    #    이후 이 파일을 기반으로 stock_recommendations.csv 생성
    check_file_date_and_execute(stock_data_file, recommendation_algorithm)

    # 2. 네이버 금융 최근 조회 상위 종목 크롤링 및 last_searched_stocks.csv 생성 (오늘 날짜 파일 없으면 실행)
    check_file_date_and_execute(last_searched_file, crawl_naver_finance_last_search)

    # 사용할 추천 데이터 파일 리스트
    # recommendation_algorithm() 함수가 stock_recommendations.csv를 생성하므로, 이 파일이 있어야 함
    recommendation_files = []
    if Path(recommendation_output_file).exists():
        recommendation_files.append(recommendation_output_file)
    else:
        print(f"경고: {recommendation_output_file} 파일이 없습니다. 이 파일은 추천 소스에서 제외됩니다.")
    
    if Path(last_searched_file).exists():
        recommendation_files.append(last_searched_file)
    else:
        print(f"경고: {last_searched_file} 파일이 없습니다. 이 파일은 추천 소스에서 제외됩니다.")

    if not recommendation_files:
        print("사용할 수 있는 추천 데이터 파일이 없습니다. 추천을 진행할 수 없습니다.")
    else:
        # 추천받고 싶은 종목 리스트
        target_stocks_for_recommendation = ['삼성전자'] # 예시 종목

        # 여러 종목에 대한 추천 결과 출력 (결과는 JSON 문자열이지만, print_unique_recommendations 내부에서 파싱하여 사용)
        print_unique_recommendations(recommendation_files, target_stocks_for_recommendation, num_recommendations=5)

        # 단일 종목에 대한 JSON 결과 직접 확인 (get_unique_recommendations의 반환값)
        # print("\n--- 단일 종목 JSON 결과 테스트 ---")
        # single_stock_name = '삼성전자'
        # json_output_single = get_unique_recommendations(recommendation_files, single_stock_name)
        # print(f"'{single_stock_name}' 추천 결과 (JSON): {json_output_single}")
        
        # single_stock_name_non_exist = '없는종목XYZ'
        # json_output_single_non_exist = get_unique_recommendations(recommendation_files, single_stock_name_non_exist)
        # print(f"'{single_stock_name_non_exist}' 추천 결과 (JSON): {json_output_single_non_exist}")

        # get_multiple_stocks_recommendations의 직접적인 JSON 반환값 확인
        # print("\n--- 다수 종목 JSON 결과 테스트 (직접호출) ---")
        # json_output_multiple = get_multiple_stocks_recommendations(recommendation_files, ['카카오', 'LG에너지솔루션'])
        # print(f"카카오, LG에너지솔루션 추천 결과 (JSON): {json_output_multiple}")