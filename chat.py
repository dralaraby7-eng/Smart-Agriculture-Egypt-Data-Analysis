import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from openai import OpenAI

# --- 1. الإعدادات الأمنية والتقنية ---
# استبدل المفتاح أدناه بمفتاح OpenRouter الخاص بك
OPENROUTER_API_KEY = "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" 

st.set_page_config(
    page_title="محلل بيانات الزراعة في مصر (2000-2025)",
    page_icon="🌾",
    layout="wide"
)

# تخصيص المظهر (CSS) لجعل الواجهة أكثر احترافية
# تخصيص المظهر (CSS) لجعل الواجهة أكثر احترافية
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    /* تعديل لون زر الإرسال */
    .stButton>button { 
        width: 100%; 
        border-radius: 5px; 
        height: 3em; 
        background-color: #2e7d32; 
        color: white; 
    }
    /* تحسين شكل حقول الإدخال */
    .stTextInput>div>div>input { 
        border-radius: 5px; 
    }
    </style>
    """, unsafe_allow_html=True)  

# --- 2. تحميل ومعالجة البيانات ---
@st.cache_data
def get_data():
    try:
        df = pd.read_csv('final.csv')
        df.columns = df.columns.str.strip() 
        
        if 'Year Code' in df.columns:
            df['Year Code'] = pd.to_numeric(df['Year Code'], errors='coerce')
            df = df.dropna(subset=['Year Code']) # حذف الصفوف التي ليس بها سنة
            df['Year Code'] = df['Year Code'].astype(int)
        return df
    except Exception as e:
        st.error(f"خطأ في تحميل الملف: {e}")
        return None

df_raw = get_data()

# --- 3. الواجهة الجانبية (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1547/1547183.png", width=80)
    st.title("نظام تحليل البيانات")
    st.write("---")
    
    # اختيار المدى الزمني بناءً على "Year Code"
    year_range = st.slider(
        "نطاق السنوات المطلوب تحليلها:",
        min_value=2000,
        max_value=2025,
        value=(2000, 2025),
        help="سيقوم البوت بتحليل البيانات فقط ضمن هذا النطاق"
    )
    
    st.success(f"تم تحميل البيانات للفترة المحددة")
    st.caption("تم تطويره لخدمة قطاع تحليل البيانات الزراعية بمصر.")

# --- 4. إعداد محرك الذكاء الاصطناعي ---
if df_raw is not None:
    # فلترة البيانات بناءً على العمود الجديد Year Code
    df_filtered = df_raw[(df_raw['Year Code'] >= year_range[0]) & (df_raw['Year Code'] <= year_range[1])]

    # إعداد الموديل بشكل مخفي عن المستخدم
    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o", 
        temperature=0
    )

    # تعليمات الخبير (System Prompt)
    system_instruction = f"""
    أنت الآن 'المحلل الزراعي الذكي'. مهمتك الإجابة على استفسارات المستخدم بناءً على ملف 'final.csv'.
    سياق العمل الحالي:
    - العمود المسؤول عن السنوات هو 'Year Code'.
    - نطاق البحث الحالي للمستخدم هو من عام {year_range[0]} إلى {year_range[1]}.
    - يجب أن تقوم بتحليل الاتجاهات (Trends) والنمو قبل كتابة الرد.
    - إذا طلب المستخدم إحصائية، ابحث عنها في 'Year Code' المناسب.
    - لغة الحوار: العربية الفصحى المبسطة بأسلوب مهني.
    """

    # إنشاء العميل الذكي للتعامل مع Pandas
    agent = create_pandas_dataframe_agent(
        llm, 
        df_filtered, 
        verbose=False, 
        allow_dangerous_code=True,
        agent_type="openai-tools",
        suffix=system_instruction
    )

    # --- 5. واجهة المحادثة (Chat Interface) ---
    st.title("📊 مساعد قرارات الزراعة المصرية")
    st.subheader(f"تحليل ذكي للفترة من {year_range[0]} إلى {year_range[1]}")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "أنا جاهز لتحليل بيانات الزراعة في مصر. يمكنك سؤالي عن الإنتاجية، المساحات المزروعة، أو مقارنات بين السنين."}
        ]

    # عرض الرسائل
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # منطقة الإدخال
    if prompt := st.chat_input("مثال: ما هو أكثر محصول تم إنتاجه في عام 2020؟"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("جاري قراءة البيانات وتحليل الإحصائيات..."):
                try:
                    # توجيه السؤال للوكيل
                    response = agent.invoke(prompt)
                    output = response["output"]
                    st.markdown(output)
                    st.session_state.messages.append({"role": "assistant", "content": output})
                except Exception as e:
                    st.error("حدث خطأ تقني في تحليل البيانات. حاول إعادة صياغة السؤال.")
else:
    st.warning("يرجى وضع ملف 'final.csv' في نفس مسار الكود ليعمل النظام.")
    
    