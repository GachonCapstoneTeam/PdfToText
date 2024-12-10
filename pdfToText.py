import fitz  # PyMuPDF
import logging
from typing import List, Optional, Tuple, Dict, Union

class PDFTextExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def extract_text(
        self, 
        page_number: int, 
        rect_coordinates: Tuple[float, float, float, float],
        stop_keywords: Optional[List[Dict[str, Union[str, bool]]]] = None
    ) -> str:
        try:
            # PDF 열기
            doc = fitz.open(self.pdf_path)
            
            # 페이지 가져오기
            page = doc[page_number]
            
            # 영역 지정
            rect = fitz.Rect(*rect_coordinates)
            
            # 해당 영역에서 텍스트 추출
            text = page.get_text("text", clip=rect)
            
            # 문서 닫기
            doc.close()

            # 줄 단위로 분리
            lines = text.split('\n')
            """
            # 디버그: 전체 추출된 라인 출력
            self.logger.debug("전체 추출된 라인:")
            for idx, line in enumerate(lines):
                self.logger.debug(f"{idx}: {line}")
            """

            # 중단 키워드 처리
            if stop_keywords:
                for keyword_config in stop_keywords:
                    keyword = keyword_config.get('keyword', '')
                    exclude_two_before = keyword_config.get('exclude_keyword_two_before_line', False)
                    
                    # self.logger.debug(f"키워드 처리: '{keyword}', exclude_two_before: {exclude_two_before}")
                    
                    for i, line in enumerate(lines):
                        if keyword in line.strip():
                            # self.logger.debug(f"키워드 발견: '{keyword}' in line {i}: {line}")
                            
                            # exclude_keyword_two_before_line 옵션 처리
                            if exclude_two_before:
                                # 키워드 2줄 전까지 텍스트 추출
                                cut_index = max(0, i - 2)
                                text = '\n'.join(lines[:cut_index])
                            else:
                                # 키워드 줄 전까지 텍스트 추출
                                text = '\n'.join(lines[:i])
                            
                            break
            
            return text.strip()
        
        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
            return ""

# 증권사별 기본 설정 딕셔너리
SECURITIES_CONFIGS = {
    "SK증권": {
        "page_num": 0,
        "coordinates": (180, 150, 700, 690),
        "stop_keywords": []
    },
    "교보증권": {
        "page_num": 0,
        "coordinates": (190, 240, 700, 655),
        "stop_keywords": []
    },
    "나이스디앤비": {
        "page_num": 1,
        "coordinates": (180, 200, 700, 682),
        "stop_keywords": []
    },
    "메리츠증권": {
        "page_num": 0,
        "coordinates": (210, 200, 700, 700),
        "stop_keywords": [{"keyword": "EPS (원)", "exclude_keyword_two_before_line": False}]
    },
    "미래에셋증권": {
        "page_num": 0,
        "coordinates": (200, 170, 700, 650),
        "stop_keywords": []
    },
    "삼성증권": {
        "page_num": 0,
        "coordinates": (200, 170, 700, 700),
        "stop_keywords": [{"keyword": "분기 실적", "exclude_keyword_two_before_line": False}]
    },
    "신한투자증권": {
        "page_num": 0,
        "coordinates": (20, 190, 350, 560),
        "stop_keywords": []
    },
    "유안타증권": {
        "page_num": 0,
        "coordinates": (30, 180, 400, 600),
        "stop_keywords": []
    },
    "유진투자증권": {
        "page_num": 0,
        "coordinates": (30, 275, 680, 700),
        "stop_keywords": [{"keyword": "시가총액(십억원)", "exclude_keyword_two_before_line": True}]
    },
    "키움증권": {
        "page_num": 0,
        "coordinates": (220, 190, 700, 850),
        "stop_keywords": []
    },
    "하나증권": {
        "page_num": 0,
        "coordinates": (179, 140, 700, 850),
        "stop_keywords": []
    },
    "한국IR협의회": {
        "page_num": 1,
        "coordinates": (40, 200, 370, 700),
        "stop_keywords": [{"keyword": "Forecast earnings & Valuation", "exclude_keyword_two_before_line": False}]
    },
    "한국기술신용평가(주)": {
        "page_num": 1,
        "coordinates": (180, 200, 700, 682),
        "stop_keywords": []
    },
    "한화투자증권": {
        "page_num": 0,
        "coordinates": (240, 220, 680, 800),
        "stop_keywords": []
    },
}

def extract_report_text(pdf_path: str, securities_firm: str) -> str:
    """
    특정 증권사 리포트에서 텍스트 추출
    
    :param pdf_path: PDF 파일 경로
    :param securities_firm: 증권사 이름
    :return: 추출된 텍스트
    """
    config = SECURITIES_CONFIGS.get(securities_firm)
    
    if not config:
        logging.warning(f"{securities_firm}에 대한 설정이 없습니다.")
        return ""
    
    extractor = PDFTextExtractor(pdf_path)
    return extractor.extract_text(
        page_number=config.get('page_num', 1),
        rect_coordinates=config.get('coordinates', (0, 0, 0, 0)),
        stop_keywords=config.get('stop_keywords', [])
    )

# 사용 예시
if __name__ == "__main__":
    pdf_file_path = "20241206_company_651563000.pdf"
    securities_firm = "유진투자증권"  # 원하는 증권사 이름 입력
    
    text = extract_report_text(pdf_file_path, securities_firm)
    print(f"{securities_firm} 리포트 텍스트:")
    print(text)