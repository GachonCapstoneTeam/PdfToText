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

def clean_change_value(value):
    direction = "+" if "상승" in value else "-" if "하락" in value else ""
    number = ''.join(filter(str.isdigit, value))
    return f"{direction}{number}" if number else "0"

def clean_stock_name(name):
    return ' '.join(name.split())

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
        time.sleep(1)
        
        df = crawl_industry_details(industry_url, industry_name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            all_data.append(df)
            print(f"{industry_name} 업종: {len(df)}개 종목 수집 완료")
    
    if all_data:
        # 모든 데이터를 하나의 DataFrame으로 합치기
        final_df = pd.concat(all_data, ignore_index=True)
        
        # CSV 파일로 저장
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
    
    numeric_columns = ["현재가", "전일거래량", "등락률"]
    for col in numeric_columns:
        df[col] = df[col].apply(convert_to_numeric)
    
    return df

def calculate_similarity(df, num_recommendations=10):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[["전일거래량", "등락률"]] = scaler.fit_transform(df_scaled[["전일거래량", "등락률"]])
    
    distance_matrix = cdist(df_scaled[["전일거래량", "등락률"]], df_scaled[["전일거래량", "등락률"]], metric="euclidean")
    
    recommendations = {}
    for idx, stock in enumerate(df["종목명"]):
        industry = df.loc[idx, "업종명"]
        distances = distance_matrix[idx]
        
        # 같은 업종 내에서 추천 종목 찾기
        same_industry_indices = df[df["업종명"] == industry].index.tolist()
        same_industry_recs = [i for i in np.argsort(distances) if i in same_industry_indices and i != idx]
        
        # 같은 업종 추천 종목이 부족하면 다른 업종에서 추가
        if len(same_industry_recs) < num_recommendations:
            other_industry_recs = [i for i in np.argsort(distances) if i not in same_industry_recs and i != idx]
            same_industry_recs.extend(other_industry_recs)
        
        recommendations[stock] = df["종목명"].iloc[same_industry_recs[:num_recommendations]].tolist()
    
    return recommendations

def export_recommendations_to_csv(df, recommendations, output_file='stock_recommendations.csv'):
    rec_columns = [f'추천종목_{i+1}' for i in range(10)]
    
    result_df = df[['업종명', '종목명']].copy()
    
    for col in rec_columns:
        result_df[col] = ''
    
    for stock, recs in recommendations.items():
        idx = result_df[result_df['종목명'] == stock].index[0]
        result_df.loc[idx, rec_columns] = recs
    
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"추천 결과가 {output_file}에 저장되었습니다.")
    return result_df

def recommendation_algorithm():
    get_industry_details()
    file_path = "stock_data.csv"
    
    df = load_data(file_path)
    print(f"알고리즘 계산을 시작합니다")
    recommendations = calculate_similarity(df)
    
    export_recommendations_to_csv(df, recommendations)

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
            return "테이블을 찾을 수 없습니다."
        
        results = []
        
        for tr in table.find_all('tr')[1:]:
            name_td = tr.select_one('td:nth-of-type(2)')
            
            if name_td and name_td.text.strip():
                stock_name = name_td.text.replace('\n', '').replace('\t', '').strip()
                results.append([stock_name])
        
        df = pd.DataFrame(results)
        
        # CSV 파일로 저장
        df.to_csv('last_searched_stocks.csv', index=False, header=False, encoding='utf-8-sig')
        print(f"last_searched_stocks.csv에 저장되었습니다.")
        return df
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None
    
def get_random_recommendations(stock_names, num_picks=3):
    """
    주어진 종목들에 대해 각각 3개의 추천 종목을 랜덤하게 추출합니다.
    
    Parameters:
    stock_names (list): 추천을 원하는 종목명 리스트
    num_picks (int): 각 종목당 추출할 추천 종목 수 (기본값: 3)
    
    Returns:
    dict: 입력 종목별 랜덤 추천 종목 딕셔너리
    """
    # CSV 파일 읽기
    result_df = pd.read_csv('stock_recommendations.csv', encoding='utf-8-sig')
    
    random_recommendations = {}
    
    for stock in stock_names:
        # 해당 종목이 데이터프레임에 있는지 확인
        if stock not in result_df['종목명'].values:
            random_recommendations[stock] = f"'{stock}'은(는) 목록에 없는 종목입니다."
            continue
            
        # 해당 종목의 추천 종목들 가져오기
        stock_row = result_df[result_df['종목명'] == stock].iloc[0]
        recommendations = [stock_row[f'추천종목_{i+1}'] for i in range(10)]
        
        # 랜덤하게 3개 선택
        selected = random.sample(recommendations, min(num_picks, len(recommendations)))
        random_recommendations[stock] = selected
    
    return random_recommendations

# 사용 예시
def print_random_recommendations(stock_names):
    """
    선택한 종목들의 랜덤 추천 결과를 출력합니다.
    
    Parameters:
    stock_names (list): 추천을 원하는 종목명 리스트
    """
    recommendations = get_random_recommendations(stock_names)
    
    print("\n=== 종목별 추천 결과 ===")
    for stock, recs in recommendations.items():
        if isinstance(recs, str):  # 에러 메시지인 경우
            print(f"\n{recs}")
        else:
            print(f"\n{stock}의 추천 종목:")
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec}")

# 단일 종목 추천 확인
#print_random_recommendations(['삼성전자'])

# 여러 종목 동시 추천 확인
#print_random_recommendations(['삼성전자', 'SK하이닉스', 'NAVER'])

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
            return "테이블을 찾을 수 없습니다."
        
        results = []
        
        for tr in table.find_all('tr')[1:]:
            name_td = tr.select_one('td:nth-of-type(2)')
            
            if name_td and name_td.text.strip():
                stock_name = name_td.text.replace('\n', '').replace('\t', '').strip()
                results.append([stock_name])
        
        df = pd.DataFrame(results)
        
        # CSV 파일로 저장
        df.to_csv('last_searched_stocks.csv', index=False, header=False, encoding='utf-8-sig')
        
        return df
    
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None
    
def get_recommendations_for_stocks(file_path, target_stocks, num_random_recs=2):
    """
    인기 종목들에 대한 랜덤 추천 종목을 출력하는 함수
    
    Args:
        file_path (str): CSV 파일 경로
        target_stocks (list): 조회하고 싶은 종목명 리스트
        num_random_recs (int): 각 종목당 추출할 랜덤 추천 종목 수
    """
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    rec_columns = [f'추천종목_{i+1}' for i in range(10)]
    
    for stock in target_stocks:
        stock_data = df[df['종목명'] == stock]
        
        if stock_data.empty:
            print(f"'{stock}' 종목을 찾을 수 없습니다.")
            continue
            
        row = stock_data.iloc[0]
        original_stock = row['종목명']
        original_industry = row['업종명']
        
        recommended_stocks = [stock for stock in row[rec_columns] if pd.notna(stock)]
        
        if recommended_stocks:
            random_recs = random.sample(recommended_stocks, min(num_random_recs, len(recommended_stocks)))
            
            print(f"원본 종목: {original_stock} (업종: {original_industry})")
            print("랜덤 추천 종목:")
            for rec in random_recs:
                print(f"- {rec}")
            print("\n")

### 사용 예시
#file_path = 'stock_recommendations.csv'
#my_stocks = ['삼성전자', 'SK하이닉스', 'NAVER']  # 출력하는 종목명 리스트
#get_recommendations_for_stocks(file_path, my_stocks)

def check_file_date_and_execute(file_path, function_to_execute):
    """
    파일의 생성 날짜를 확인하고, 오늘 날짜가 아니면 지정된 함수를 실행합니다.
    
    Parameters:
    file_path (str): 확인할 파일의 경로
    function_to_execute (callable): 실행할 함수
    
    Returns:
    bool: 함수가 실행되었으면 True, 아니면 False
    """
    try:
        # 파일의 생성 시간 가져오기
        file_creation_time = os.path.getctime(file_path)
        creation_date = datetime.datetime.fromtimestamp(file_creation_time).date()
        
        # 오늘 날짜 가져오기
        today = datetime.date.today()
        
        # 날짜 비교
        if creation_date != today:
            print(f"파일 생성일: {creation_date}")
            print(f"오늘 날짜: {today}")
            print("파일이 오늘 생성되지 않았으므로 함수를 실행합니다.")
            function_to_execute()
            return True
        else:
            print("파일이 오늘 생성되었습니다. 함수를 실행하지 않습니다.")
            return False
            
    except FileNotFoundError:
        print(f"에러: {file_path} 파일을 찾을 수 없습니다.")
        print(f"파일을 생성합니다")
        function_to_execute()
        return True
    except Exception as e:
        print(f"에러 발생: {e}")
        return False


# 파일 경로와 함수를 지정하여 실행
file_path_1 = "stock_data.csv"
file_path_2 = "last_searched_stocks.csv"
check_file_date_and_execute(file_path_1, recommendation_algorithm)
check_file_date_and_execute(file_path_2, crawl_naver_finance_last_search)