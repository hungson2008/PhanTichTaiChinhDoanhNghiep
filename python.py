import streamlit as st
import pandas as pd
import json
import requests
import time
from io import BytesIO

# --- Cấu hình API và Model ---
# Model sử dụng cho phân tích văn bản và grounding
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
# API Key sẽ được tự động cung cấp trong môi trường Canvas nếu để trống
API_KEY = "" 

# Tên các sheet bắt buộc (theo yêu cầu của khách hàng)
# Đã đổi key sang các ký hiệu viết tắt chuẩn: CDKT, KQHDKD, BCLCTT
SHEET_NAMES = {
    'CDKT': 'Bảng Cân đối Kế toán (Balance Sheet)',
    'KQHDKD': 'Báo cáo Kết quả Hoạt động Kinh doanh (Income Statement)',
    'BCLCTT': 'Báo cáo Lưu chuyển Tiền tệ (Cash Flow Statement)'
}

# --- Hàm Helpers Xử lý Dữ liệu và API ---

def format_df_to_markdown(df, title):
    """Chuyển đổi DataFrame thành chuỗi Markdown để đưa vào Prompt."""
    # Đảm bảo DF chỉ có 2 cột: 'Chỉ tiêu' và 'Số liệu'
    if df.shape[1] < 2:
        return f"Không đủ dữ liệu trong sheet: {title}"

    # Đặt tên lại cho 2 cột đầu tiên để dễ đọc
    df.columns = ['Chỉ tiêu', 'Số liệu'] + list(df.columns[2:])
    
    # Giới hạn số lượng hàng để tránh prompt quá dài
    df_preview = df.head(50) 
    
    markdown_table = f"### {title}\n"
    markdown_table += df_preview.to_markdown(index=False)
    
    return markdown_table

def call_gemini_api_with_backoff(user_query, system_prompt, max_retries=5):
    """
    Gọi API Gemini với cơ chế Exponential Backoff và Google Search Grounding.
    """
    st.info("Đang gọi AI Gemini để phân tích rủi ro. Quá trình này có thể mất vài giây...")
    
    # Cấu trúc payload cho API
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        # Kích hoạt Google Search để grounding với thông tin mới nhất về quy định ngân hàng
        "tools": [{"google_search": {}}], 
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    headers = {'Content-Type': 'application/json'}
    
    # Xử lý URL: chỉ thêm API key nếu nó tồn tại, nếu không sẽ dựa vào môi trường Canvas để xác thực
    request_url = API_URL
    if API_KEY:
        request_url = f"{API_URL}?key={API_KEY}"
        
    for attempt in range(max_retries):
        try:
            response = requests.post(
                request_url, # Sử dụng request_url đã được xây dựng
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status() # Lỗi HTTP sẽ ném ra exception
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            # Trích xuất nội dung và nguồn grounding
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
            
            sources = []
            grounding_metadata = candidate.get('groundingMetadata')
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = grounding_metadata['groundingAttributions']
                
            return text, sources
            
        except requests.exceptions.RequestException as e:
            # Bắt lỗi 403 cụ thể (thường là lỗi xác thực)
            if 'response' in locals() and response.status_code == 403:
                st.error("Lỗi 403 Forbidden: Lỗi xác thực (API Key). Vui lòng kiểm tra lại cấu hình môi trường hoặc API Key.")
                return f"Đã thất bại: Lỗi xác thực (403 Forbidden).", []

            st.warning(f"Lỗi API (lần {attempt + 1}): {e}. Đang thử lại...")
            if attempt < max_retries - 1:
                # Dừng lại theo cấp số nhân (1s, 2s, 4s, ...)
                time.sleep(2 ** attempt) 
            else:
                return f"Đã thất bại sau {max_retries} lần thử: Không thể kết nối với dịch vụ AI.", []
        except Exception as e:
             return f"Lỗi không xác định: {e}", []

# --- Logic Ứng dụng Streamlit ---

st.set_page_config(
    page_title="Phân tích Tài chính Doanh nghiệp (AI-Powered)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Ứng dụng Phân tích Rủi ro Tín dụng Doanh nghiệp")
st.markdown("Sử dụng Gemini AI và Google Search Grounding để phân tích Báo cáo Tài chính của khách hàng theo chuẩn Ngân hàng.")

# Khởi tạo state để lưu trữ kết quả
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'sources' not in st.session_state:
    st.session_state.sources = []

uploaded_file = st.file_uploader(
    "Tải lên file Báo cáo Tài chính (Excel .xlsx)", 
    type=["xlsx"]
)

if uploaded_file is not None:
    st.success(f"File **{uploaded_file.name}** đã được tải lên.")
    
    # Đọc tất cả các sheet vào một dictionary
    try:
        xls = pd.ExcelFile(uploaded_file)
        all_sheets = {sheet: xls.parse(sheet).dropna(how='all') for sheet in xls.sheet_names}
    except Exception as e:
        st.error(f"Lỗi khi đọc file Excel: {e}")
        st.stop()

    # Xác định các sheet cần thiết và chuẩn bị dữ liệu
    financial_data = {}
    missing_sheets = []
    
    for key, description in SHEET_NAMES.items():
        # Tìm kiếm sheet không phân biệt hoa thường, cho phép người dùng đặt tên sheet chứa 'cdkt' hoặc 'kqhkd'...
        found_sheet_name = next((sheet for sheet in all_sheets if key.lower() in sheet.lower()), None)
        
        if found_sheet_name:
            df = all_sheets[found_sheet_name]
            financial_data[key] = format_df_to_markdown(df, description)
        else:
            missing_sheets.append(description)

    if missing_sheets:
        st.warning(f"⚠️ Thiếu các sheet bắt buộc: **{', '.join(missing_sheets)}**. Vui lòng kiểm tra lại tên sheet.")
        st.stop()
    
    # --- Xây dựng Prompt và Hệ thống ---
    
    # 1. System Instruction (Hướng dẫn vai trò cho AI)
    system_prompt = (
        "Bạn là một chuyên gia phân tích tín dụng và tài chính doanh nghiệp hàng đầu, có kinh nghiệm sâu sắc về các quy định cho vay của ngành ngân hàng Việt Nam. "
        "Nhiệm vụ của bạn là phân tích chi tiết các Báo cáo Tài chính (Cân đối kế toán, Kết quả kinh doanh, Lưu chuyển tiền tệ) được cung cấp. "
        "Dựa trên phân tích, bạn phải đưa ra một đánh giá tổng quan về tình hình tài chính của doanh nghiệp và **chỉ ra 3 đến 5 rủi ro trọng yếu nhất** mà ngân hàng cần xem xét khi quyết định cho vay. "
        "Đặc biệt chú trọng đến các chỉ số thanh khoản, đòn bẩy, khả năng sinh lời và dòng tiền. "
        "Phản hồi phải bằng tiếng Việt, có cấu trúc rõ ràng với các phần: **1. Đánh giá Tổng quan** và **2. Các Rủi ro Trọng yếu** (sử dụng dấu gạch đầu dòng)."
    )
    
    # 2. User Query (Dữ liệu đầu vào cho AI)
    user_query = "Dưới đây là Dữ liệu Báo cáo Tài chính của khách hàng. Hãy thực hiện phân tích và nhận diện rủi ro theo hướng dẫn:\n\n"
    
    for key, data_markdown in financial_data.items():
        user_query += f"{data_markdown}\n\n"
        
    st.subheader("Dữ liệu đã sẵn sàng để phân tích:")
    st.code(f"Kích thước Prompt: {len(user_query)} ký tự. (Chỉ 50 dòng đầu tiên của mỗi sheet được gửi)", language='text')

    
    if st.button("🚀 Phân tích Rủi ro Tín dụng bằng AI", type="primary"):
        with st.spinner('Đang phân tích sâu...'):
            st.session_state.analysis_result, st.session_state.sources = call_gemini_api_with_backoff(
                user_query, 
                system_prompt
            )
        st.success("Phân tích hoàn tất!")

# --- Hiển thị Kết quả ---

if st.session_state.analysis_result:
    st.divider()
    st.header("Kết quả Phân tích Rủi ro từ AI")
    st.markdown(st.session_state.analysis_result)

    if st.session_state.sources:
        st.subheader("Nguồn tham khảo (Grounding)")
        st.markdown("AI đã tham khảo thông tin cập nhật từ các nguồn sau để đảm bảo tính chính xác theo quy định:")
        
        source_markdown = ""
        for i, source in enumerate(st.session_state.sources):
            if source.get('title') and source.get('uri'):
                source_markdown += f"- [{source['title']}]({source['uri']})\n"
        st.markdown(source_markdown)
    else:
        st.info("Không có nguồn Grounding nào được trích dẫn (do yêu cầu phân tích dữ liệu chuyên sâu, không phải tìm kiếm thông tin).")
elif uploaded_file is None:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
