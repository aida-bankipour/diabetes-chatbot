from flask import Flask, render_template, request, jsonify, session
import numpy as np
import os
import logging
import re
from datetime import datetime
import tensorflow as tf
import google.generativeai as genai
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)  # برای مدیریت session

# Configure Gemini API
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    logging.error(f"Error configuring Gemini API: {e}. Please ensure GEMINI_API_KEY is set.")

# Load MLP model
try:
    model_path = 'mlp_model/mlp_model.keras'
    model = tf.keras.models.load_model(model_path)
    logging.info("MLP model loaded successfully")
except Exception as e:
    logging.error(f"Error loading MLP model: {e}")
    model = None

# User state storage
user_data = {}

# Symptom keywords
symptom_keywords = {
    "پرادراری": [
        "پرادراری", "ادرار زیاد", "ادرار بیش از حد", "زیاد دستشویی می‌رم", "دستشویی رفتن زیاد",
        "شب‌ها بیدار می‌شم برای ادرار", "تکرر ادرار", "تکررادرار", "زیاد ادرار می‌کنم", "دسشویی", "Polyuria",
        "شبا چندبار دستشویی می‌رم", "مدام دستشویی لازم دارم", "ادرارم زیاده", "ادرار", "دستشویی", "دفع ادرار", "جیش", "دسشوییم", "ادرارم"
    ],
    "عطش": [
        "عطش", "تشنگی", "خیلی تشنه‌ام", "مدام آب می‌خورم", "زیاد آب می‌خورم", "پرنوشی", "استسقاء", "آب", "خشکی دهن", "خشکی زبان", "Polydipsia",
        "همیشه تشنه‌ام", "دهنم خشک می‌شه", "خشکی دهان", "تشنگی شدید", "نمی‌تونم تشنگی‌مو کنترل کنم", "استسقا", "تشنه", "پرنوشی"
    ],
    "کاهش وزن": [
        "کاهش وزن", "افت وزن", "وزنم کم شده", "بدون دلیل وزن کم کردم", "لاغر شدم", "لاغر", "کم شدن وزن",
        "وزنم داره کم می‌شه", "وزن کم کردن بی‌دلیل", "اخیرا لاغر شدم", "وزنم یهو کم شده", "وزن کم کردم"
    ],
    "ضعف": [
        "ضعف", "بی‌حالی", "انرژی ندارم", "همیشه خسته‌ام", "احساس ضعف", "زود خسته می‌شم", "خسته", "کسل", "کسلی", "خستگی", "بی حال", "خواب ", "خوابم",
        "خستگی زیاد", "بی‌جونم", "ناتوانی", "احساس خستگی مداوم", "قدرت ندارم", "خسته و کسل"
    ],
    "پرخوری": [
        "پرخوری", "زیاد می‌خورم", "اشتهام زیاد شده", "گرسنگی مداوم دارم", "همیشه گرسنه‌ام", "پراشتهایی", "اشتها", "میل", "شیرین", "شیرینی", "بخورم", "میخورم",
        "اشتهای زیادی دارم", "نمی‌تونم جلوی خوردنم رو بگیرم", "مدام دلم غذا می‌خواد", "اشتهام", "میلم", "خوردن"
    ],
    "عفونت قارچی": [
        "عفونت قارچی", "عفونت در ناحیه تناسلی", "سوزش یا خارش ناحیه تناسلی", "عفونت واژن", "قارچ", "خارش ناحیه تناسلی", "بوی بد",
        "خارش تناسلی", "سوزش هنگام ادرار", "عفونت مکرر قارچی", "قارچ پوستی"
    ],
    "تاری دید": [
        "تاری دید", "کاهش میدان دید", "چشمام تار می‌بینه", "دیدم خوب نیست", "دیدم تار شده", "تار", "محو", "بینایی", "دیدم", "بیناییم",
        "چشمام تار شده", "دیدم ضعیف شده", "نمی‌تونم واضح ببینم", "تاری تو دید", "چشمام درست نمی‌بینه"
    ],
    "خارش": [
        "خارش", "خشکی پوست", "خارش بدن", "پوستم می‌خاره", "خارش شدید", "پوستم خشک و خارش‌دار",
        "خارش پوست", "همه‌جای بدنم می‌خاره", "خارش مداوم", "پوستم خشک", "خارش"
    ],
    "عصبانیت": [
        "عصبانیت", "تحریک‌پذیری", "زود عصبی می‌شم", "کنترل احساسات سخت شده", "زود جوش میارم", "عصبانی", "عصبی", "پرخاشگر", "زود واکنش", "تحریک پذیر",
        "عصبی شدم", "حوصله‌ام کم شده", "تحملم کم شده", "زود از کوره در می‌رم"
    ],
    "تأخیر در بهبود": [
        "تأخیر در بهبود", "زخم‌هام دیر خوب می‌شن", "خوب نشدن زخم‌ها", "زخمام دیر جوش می‌خوره", "زخم", "ضخم",
        "جای زخم دیر خوب می‌شه", "بهبود زخم کند", "زخم‌هام نمی‌بنده"
    ],
    "فلج جزئی": [
        "فلج جزئی", "ضعف عضلانی", "عضلاتم ناتوان شدن", "ناتوانی در حرکت", "عضله‌هام ضعیف شدن", 
        "نمی‌تونم راحت حرکت کنم", "ضعف در عضلات", "حرکت کردن برام سخت شده"
    ],
    "درد عضلانی": [
        "درد عضلانی", "کشیدگی عضلات", "بدنم درد می‌کنه", "عضلاتم درد داره", "درد تو بدنم", "گرفتگی", "گرفتگی بدن",
        "بدن‌درد", "عضله‌هام درد می‌کنه", "درد ماهیچه‌ای", "سفتی عضلات", "خشکی عضلات", "عضلاتم گرفته", "گرفتگی عضلات", "درد عضلانی",
        "عضلاتم سفت شدن", "گرفتگی ماهیچه", "عضله‌هام منقبض شده"
    ],
    "ریزش مو": [
        "ریزش مو", "کم‌پشت شدن مو", "موهام میریزه", "موهای سرم کم شده", "ریزش موی شدید", "مو",
        "موهام داره می‌ریزه", "کم شدن مو", "طاسی"
    ],
    "چاقی": [
        "چاقی", "اضافه وزن", "خیلی چاق شدم", "وزنم رفته بالا", "وزن زیاد کردم", "چاق", "سنگین وزن",
        "چاق شدم", "وزنم بالاست", "اضافه وزن دارم"
    ],
    "قند خون بالا": [
        r"قند\s*(خون)?\s*(ناشتا(?:م)?)?(?:م)?\s*(\d+)", r"قند\s*(خون)?(?:م)?\s*بالا\s*(\d+)",
        "قند خونم بالاست", "قندم بالاست", "قند خون بالا دارم"
    ]
}

# Symptom names in order of structured questions
symptom_names = [
    "پرادراری", "عطش", "کاهش وزن", "ضعف", "پرخوری", "عفونت قارچی", "تاری دید",
    "خارش", "عصبانیت", "تأخیر در بهبود", "فلج جزئی", "درد عضلانی", "ریزش مو", "چاقی"
]

# Keywords
positive_keywords = ["بله", "آره", "اره", "دارم", "بعضی وقتا", "گاهی", "اکثرا", "همیشه", 
    "میکنم", "احساس میکنم", "شده", "پیش میاد", "زیاد", "تا حدودی", "درگیرم",
    "برام پیش اومده", "مشاهده کردم", "دیدم", "می‌ شوم", "احساس می‌کنم", "دچارم", "هستم"]
negative_keywords = ["نه", "خیر", "ندارم", "نمی‌کنم", "نیستم", "نمی‌شوم", "نمی‌خورم", "نمی‌رم", 
    "اصلا", "نداشتم", "هرگز", "کم", "خیلی کم", "نادره", "به ندرت", "تقریباً نه"]
goodbye_keywords = ["خداحافظ", "خدانگهدار", "بای", "بای بای", "بعدا می‌بینمت"]
thanks_keywords = ["ممنون", "ممنونم", "تشکر", "متشکرم"]
question_indicators = [
    "چیه", "چیست", "توضیح", "درباره", "چطور", "چگونه", "علائم", "علامت", "نشانه", "آیا",
    "چه", "کجا", "از کجا", "باید چی", "چند", "چقدر", "چگونه", "چرا", "کی", "کدام", "یعنی چی", 
    "چقدره ", " چقدر است","?","؟","نمی‌دونم", "نمیدونم", "نمی‌فهمم", "نمیفهمم"
]
test_intent_keywords = [
    "تست دیابت", "دیابت دارم", "بررسی دیابت", "تشخیص دیابت", "می‌خوام تست کنم",
    "دیابت نوع", "آزمایش دیابت"
]
invalid_response_keywords = ["نع","بلخ", "نچ"]

# Reset user state
def reset_user_state(user_id):
    user_data[user_id] = {
        "age": None,
        "gender": None,
        "symptoms": [],
        "fasting_blood_sugar": None,
        "waiting_for_questions": False,
        "current_question_index": 0,
        "current_symptoms": [],
        "prediction_done": False,
        "questions": [
            "آیا بیش از حد معمول ادرار می‌کنید؟",
            "آیا احساس تشنگی مداوم دارید؟",
            "آیا کاهش وزن ناگهانی داشته‌اید؟",
            "آیا ضعف بدنی دارید؟",
            "آیا اشتهای شما به طور غیرعادی افزایش پیدا کرده است؟",
            "آیا مبتلا به عفونت‌های قارچی هستید؟",
            "آیا تاری دید دارید؟",
            "آیا احساس خشکی یا خارش پوست دارید؟",
            "آیا به سرعت عصبی می‌شوید؟",
            "آیا بهبود زخم‌های بدنتان به کندی صورت می‌گیرد؟",
            "آیا فلج جزئی (ضعف یا کاهش توانایی حرکتی) دارید؟",
            "آیا در فعالیت‌های روزمره احساس کشیدگی یا درد عضلانی دارید؟",
            "آیا ریزش مو دارید؟",
            "آیا اضافه وزن دارید؟"
        ],
        "previous_symptoms": [],
        "expecting_age": False  # پرچم برای انتظار سن
    }
    logging.info(f"Reset user state: {user_id}")

# Gemini API response
def get_gemini_response(user_message, context="general", user_id=None):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        if context == "symptom_explanation":
            prompt = (
                "شما یک چت‌بات تشخیص اولیه دیابت هستید که به زبان فارسی پاسخ می‌دهید. "
                "به سؤال کاربر پاسخ کوتاه، دقیق و کاربرپسند بدهید. "
                "سپس از کاربر بخواهید با بله یا خیر به سؤال اصلی پاسخ دهد. "
                f"سؤال کاربر: {user_message}\n"
                f"سؤال اصلی: {user_data[user_id]['questions'][user_data[user_id]['current_question_index']]}"
            )
        else:
            previous_symptoms = user_data[user_id].get("previous_symptoms", []) if user_id else []
            prompt = (
                "شما یک چت‌بات تشخیص اولیه دیابت هستید که به زبان فارسی پاسخ می‌دهید. "
                "به سؤال کاربر پاسخ کوتاه، دقیق و کاربرپسند بدهید. "
                "اگر سؤال یا علائم نامرتبط با دیابت است (مثل سردرد، حالت تهوع)، توضیح دهید که این علائم ممکن است به دیابت ربطی نداشته باشند و پیشنهاد دهید برای بررسی بیشتر به پزشک مراجعه کنند یا علائم دیابت (مثل پرادراری، تشنگی) یا قند خون ناشتا را ارائه کنند. "
                "برای سؤالات عمومی درباره دیابت (مثل 'چه آزمایشی بدم؟' یا 'چطور مطمئن شم دیابت دارم؟')، پاسخ دقیق و مرتبط بدهید (مثل توصیه به آزمایش HbA1c یا قند خون ناشتا). "
                "مثال ورودی: '30 سال، آقا، پرادراری' یا 'قند خون ناشتا 120' یا 'دیابت ارثی است؟'. "
                f"علائم قبلی کاربر: {', '.join(previous_symptoms) if previous_symptoms else 'هیچ'}\n"
                f"سؤال کاربر: {user_message}"
            )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return "متأسفم، نمی‌توانم الان پاسخی بدهم. لطفاً سن، جنسیت، علائم (مثل پرادراری) یا قند خون‌تان را بگویید."

# Predict diabetes probability
def predict_diabetes(input_data):
    try:
        prediction = model.predict(input_data, verbose=0)
        probability = prediction[0][0] * 100
        return probability
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return 0

# Diabetes prediction response
def predict_diabetes_response(data, detailed=False):
    age = data["age"]
    gender = data["gender"]
    fasting_blood_sugar = data["fasting_blood_sugar"]
    
    # Check for hypoglycemia
    if fasting_blood_sugar is not None and fasting_blood_sugar < 70:
        return (
            f"قند خون {fasting_blood_sugar} میلی‌گرم در دسی‌لیتر خیلی پایین است (هیپوگلیسمی). "
            "لطفاً سریع یک منبع قندی (مثل آب‌میوه) مصرف کنید و ۱۵ دقیقه بعد قند خون خود را چک کنید. "
            "در صورت عدم بهبود، فوراً به پزشک مراجعه کنید."
        )

    if detailed:
        symptoms = data["current_symptoms"]
    else:
        symptoms = [0] * len(symptom_names)
        for symptom in data["symptoms"]:
            if symptom in symptom_names:
                symptoms[symptom_names.index(symptom)] = 1

    input_features = np.array([[age, gender] + symptoms], dtype=float)
    logging.info(f"Input features: {input_features}")
    probability = predict_diabetes(input_features)
    logging.info(f"Prediction probability: {probability}")

    if fasting_blood_sugar is not None and fasting_blood_sugar >= 126 or "قند خون بالا" in data["symptoms"]:
        probability = max(probability, 75)
        logging.info("Increased probability due to high blood sugar")

    if detailed:
        if probability > 50:
            return (
                "بر اساس پاسخ‌های شما، احتمال دیابت وجود دارد. چند توصیه براتان دارم:<br>"
                "- لطفاً هرچه زودتر با پزشک متخصص مشورت کنید.<br>"
                "- آزمایش‌های کامل‌تر مثل قند خون ناشتا یا HbA1c انجام بدید.<br>"
                "- رژیم غذایی خود را اصلاح کنید و مصرف قند و چربی را کم کنید.<br>"
                "- ورزش منظم (حداقل ۳۰ دقیقه در روز) را شروع کنید.<br>"
                "- اگر سابقه خانوادگی دیابت دارید، بیشتر مراقب باشید."
            )
        else:
            return (
                "بر اساس اطلاعات، خوشبختانه احتمال دیابت وجود ندارد یا حداقل پایین است. 😊<br>"
                "- سبک زندگی سالم را ادامه دهید (تغذیه متعادل و ورزش).<br>"
                "- هر چند وقت یک‌بار چکاپ منظم داشته باشید.<br>"
                "- استرس را مدیریت کنید و خواب کافی داشته باشید."
            )
    else:
        if probability > 50:
            return (
                "با توجه به اطلاعات شما، احتمال دیابت وجود دارد. "
                "برای بررسی دقیق‌تر، لطفاً کلمه «سوال» را وارد کنید تا تست کامل‌تری انجام دهیم."
            )
        else:
            if data["symptoms"] or data["fasting_blood_sugar"] is not None:
                return (
                    "احتمال ابتلا به دیابت در شما پایین است. "
                    "برای اطمینان بیشتر، می‌توانید با وارد کردن کلمه «سوال» در یک تست دقیق‌تر شرکت کنید."
                )
            else:
                return (
                    "احتمال ابتلا به دیابت در شما پایین است. "
                    "اگه علائم خاصی (مثل پرادراری یا عطش) دارید، لطفاً بگویید یا کلمه «سوال» را برای برسی دقیق تر وارد کنید."
                )

# Routes
@app.route("/")
def home():
    session.clear()  # ریست session با هر رفرش صفحه
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["message"].strip()
    user_id = session.get("user_id", str(uuid.uuid4()))
    session["user_id"] = user_id  # ذخیره user_id در session
    
    if user_id not in user_data:
        reset_user_state(user_id)
    
    logging.info(f"User input: {user_message}")
    response = process_user_input(user_message, user_id)
    return jsonify({"response": response})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

# Process user input
def process_user_input(user_input, user_id):
    current_data = user_data[user_id]
    user_input_clean = user_input.lower().replace("‌", "")
    responses = []

    # 1. Check for goodbye
    if any(word == user_input_clean for word in goodbye_keywords):
        logging.info("Detected goodbye")
        reset_user_state(user_id)
        return "خدانگهدار! امیدوارم تونسته باشم کمکتون کنم. 😊"

    # 2. Check for thanks
    if any(word in user_input_clean for word in thanks_keywords):
        logging.info("Detected thanks")
        if any(word in user_input_clean for word in goodbye_keywords):
            reset_user_state(user_id)
            return "خدانگهدار! خوشحال میشوم باز هم بتوانم کمکتان کنم. 😊"
        return "خواهش می‌کنم! اگر سوال دیگری دارید یا خواستید موضوع دیگری را بررسی کنیم، من آماده هستم."

    # 3. Handle structured question responses
    if current_data.get("waiting_for_questions", False):
        logging.info("Processing structured question response")
        current_question_index = current_data["current_question_index"]
        
        # Check for symptom explanation or general question during structured questions
        if any(indicator in user_input_clean for indicator in question_indicators):
            logging.info("Forwarding symptom explanation to Gemini API")
            gemini_response = get_gemini_response(user_input, context="symptom_explanation", user_id=user_id)
            return gemini_response

        # Check for valid responses
        if any(word in user_input_clean for word in positive_keywords):
            logging.info("Positive response to structured question")
            current_data["current_symptoms"].append(1)
            current_data["current_question_index"] += 1
        elif any(word in user_input_clean for word in negative_keywords):
            logging.info("Negative response to structured question")
            current_data["current_symptoms"].append(0)
            current_data["current_question_index"] += 1
        elif any(word in user_input_clean for word in invalid_response_keywords):
            logging.info(f"Invalid response to structured question: {user_input}")
            return f"لطفاً با بله یا خیر پاسخ دهید: {current_data['questions'][current_question_index]}"
        else:
            logging.info(f"Unrecognized response to structured question: {user_input}")
            return f"لطفاً با بله یا خیر پاسخ دهید: {current_data['questions'][current_question_index]}"

        if current_data["current_question_index"] < len(current_data["questions"]):
            return current_data["questions"][current_data["current_question_index"]]
        else:
            logging.info("Finished structured questions")
            logging.info(f"Current symptoms: {current_data['current_symptoms']}")
            current_data["waiting_for_questions"] = False
            prediction_result = predict_diabetes_response(current_data, detailed=True)
            current_data["symptoms"] = [
                symptom_names[i] for i, val in enumerate(current_data["current_symptoms"]) if val == 1
            ]
            current_data["previous_symptoms"] = current_data["symptoms"].copy()
            reset_user_state(user_id)
            return prediction_result

    # 4. Extract information
    logging.info("Extracting information")
    info_detected = False
    symptoms_detected = []
    unrelated_symptoms = []

    # Fasting blood sugar
    fbs_match = re.search(r'قند\s*(?:خون)?\s*(?:ناشتا(?:ی من|م)?)?\s*(?:من|م)?\s*(\d{2,3})|قند\b.*?\b(\d+)\b', user_input_clean)
    if fbs_match:            
        fbs_value = int(fbs_match.group(1) or fbs_match.group(2))
        current_data["fasting_blood_sugar"] = fbs_value
        info_detected = True
        logging.info(f"Detected fasting blood sugar: {fbs_value}")
        if fbs_value < 70:
            responses.append(
                f"قند خون {fbs_value} میلی‌گرم در دسی‌لیتر خیلی پایین است (هیپوگلیسمی). "
                "لطفاً سریعا یک منبع قندی (مثل آب‌میوه) مصرف کنید و ۱۵ دقیقه بعد قند خون‌تان را مجدد چک کنید."
            )
        elif fbs_value >= 100 and fbs_value < 126:
            responses.append(
                f"قند خون ناشتای {fbs_value} میلی‌گرم در دسی‌لیتر در محدوده پیش‌دیابت قرار دارد. "
                "این یعنی ممکن است در معرض خطر دیابت باشید."
            )
        elif fbs_value >= 126 and "قند خون بالا" not in current_data["symptoms"]:
            current_data["symptoms"].append("قند خون بالا")
            logging.info("Added symptom: قند خون بالا")
            responses.append(f"قند خون ناشتای {fbs_value} میلی‌گرم در دسی‌لیتر بالاتر از حد نرمال است.")

    # Age
    if current_data["expecting_age"]:
        age_match = re.search(r'(\d+)\s*سال', user_input, re.IGNORECASE)
        standalone_age_match = re.search(r'\b(\d+)\b(?!\s*%)', user_input_clean)
        if age_match:
            age = int(age_match.group(1))
            if 0 <= age <= 99:
                current_data["age"] = age
                info_detected = True
                current_data["expecting_age"] = False
                logging.info(f"Detected age: {current_data['age']}")
        elif standalone_age_match:
            age = int(standalone_age_match.group(1))
            if 0 <= age <= 99:
                current_data["age"] = age
                info_detected = True
                current_data["expecting_age"] = False
                logging.info(f"Detected age: {current_data['age']}")
    else:
        age_match = re.search(r'(\d+)\s*سال', user_input, re.IGNORECASE)
        standalone_age_match = re.search(r'\b(\d+)\b(?!\s*%)', user_input_clean)
        if age_match:
            age = int(age_match.group(1))
            if 0 <= age <= 99:
                current_data["age"] = age
                info_detected = True
                logging.info(f"Detected age: {current_data['age']}")
        elif standalone_age_match:
            age = int(standalone_age_match.group(1))
            if 0 <= age <= 99:
                current_data["age"] = age
                info_detected = True
                logging.info(f"Detected age: {current_data['age']}")

    # Gender (only set if not previously set)
    if current_data["gender"] is None:
        if any(g in user_input_clean for g in ["خانم", "زن", "دختر", "مونث"]):
            current_data["gender"] = 0
            info_detected = True
            logging.info("Detected gender: خانم")
        elif any(g in user_input_clean for g in ["آقا", "مرد", "پسر", "مذکر"]):
            current_data["gender"] = 1
            info_detected = True
            logging.info("Detected gender: آقا")

    # Symptoms
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            if symptom == "قند خون بالا" and not fbs_match:
                continue
            pattern = re.compile(r'\b' + re.escape(keyword.replace(r'\d+', r'\d+')) + r'\b', re.IGNORECASE)
            if pattern.search(user_input_clean) and symptom not in current_data["symptoms"]:
                symptoms_detected.append(symptom)
                break

    # Check for unrelated symptoms
    unrelated_symptom_patterns = [
        r'سردرد', r'تهوع', r'سرگیجه', r'درد\s*شکم', r'تب', r'سرفه', r'گلودرد', r'خونریزی',
        r'کمردرد', r'پهلو\s*درد', r'فشار\s*(خون)?\s*بالا', r'دل\s*درد', r'تنگی\s*نفس',
        r'درد\s*قفسه\s*سینه', r'تپش\s*قلب', r'اسهال', r'یبوست', r'حالت\s*تهوع',
        r'درد\s*معده', r'سوزش\s*معده', r'نفخ', r'سوء\s*هاضمه', r'درد\s*مفصل',
        r'گرگرفتگی', r'لرز', r'خون\s*دماغ', r'گوش\s*درد', r'چشم\s*درد', r'گلو\s*درد',
        r'حساسیت', r'آلرژی', r'جوش\s*صورت', r'خارش\s*گلو', r'درد\s*گوش', r'فشار\s*(خون)?\s', r'درد'
        ]
    for pattern in unrelated_symptom_patterns:
        if re.search(pattern, user_input_clean):
            unrelated_symptoms.append(user_input_clean)
            break

    if symptoms_detected:
        current_data["symptoms"].extend(symptoms_detected)
        info_detected = True
        current_data["previous_symptoms"].extend(symptoms_detected)
        logging.info(f"Detected symptoms: {symptoms_detected}")

    # Check for "no symptoms"
    if "علائمی ندارم" in user_input_clean or "هیچ علامتی" in user_input_clean:
        logging.info("Detected no symptoms")
        reset_user_state(user_id)
        return "به نظر میرسد مشکلی ندارید! برای شما آرزوی سلامتی می‌کنم. 😊 اگه با علائم جدیدی روبه رو شدید، میتوانید روی کمک من حساب کنید."

    # 5. Handle general questions or test intent
    if any(indicator in user_input_clean for indicator in question_indicators) or unrelated_symptoms:
        logging.info("Detected general question or unrelated symptoms")
        gemini_response = get_gemini_response(user_input, user_id=user_id)
        responses.append(gemini_response)
        return ", ".join(responses)

    # 6. Request missing information
    if info_detected:
        missing_info = []
        if current_data["age"] is None:
            missing_info.append("سن‌تان")
            current_data["expecting_age"] = True  # Set expecting_age flag
        if current_data["gender"] is None:
            missing_info.append("جنسیت‌ خود (آقا یا خانم)")
        if not current_data["symptoms"] and current_data["fasting_blood_sugar"] is None and not unrelated_symptoms:
            missing_info.append("علائم‌تان (مثل پرادراری، تشنگی) یا قند خون ناشتا")

        if missing_info:
            logging.info(f"Requesting missing information: {', '.join(missing_info)}")
            responses.append(f"لطفاً {', '.join(missing_info)} را بگویید تا بتوانم بررسی دقیق‌تری انجام بدهم.")
            return ", ".join(responses)

    # 7. Handle test intent or structured questions
    test_intent = any(keyword in user_input_clean for keyword in test_intent_keywords)
    if user_input_clean == "سوال" or test_intent:
        if current_data["age"] is not None and current_data["gender"] is not None:
            current_data["waiting_for_questions"] = True
            current_data["current_question_index"] = 0
            current_data["current_symptoms"] = []
            logging.info("Starting structured questions")
            return current_data["questions"][0]
        else:
            logging.info("Insufficient data for structured questions")
            missing_info = []
            if current_data["age"] is None:
                missing_info.append("سن‌تان")
                current_data["expecting_age"] = True
            if current_data["gender"] is None:
                missing_info.append("جنسیت‌ خود (آقا یا خانم)")
            responses.append(f"لطفاً {', '.join(missing_info)} را بگویید تا برسی را شروع کنیم.")
            return ", ".join(responses)

    # 8. Perform prediction if all data provided
    if (current_data["age"] is not None and
        current_data["gender"] is not None and
        (current_data["symptoms"] or current_data["fasting_blood_sugar"] is not None)):
        if not current_data["prediction_done"]:
            logging.info("Performing initial prediction")
            current_data["prediction_done"] = True
            prediction_result = predict_diabetes_response(current_data)
            responses.append(f"بابت اطلاعاتی که وارد کردید سپاسگزارم. {prediction_result}")
            return ", ".join(responses)

    # 9. Handle greetings or unknown input
    if user_input_clean in ["سلام", "سلام علکیم", "سلام خوبی"]:
        return "سلام! 😊 برای بررسی دیابت، لطفاً سن، جنسیت، علائم (مثل پرادراری) یا قند خون‌ خود را بگویید (مثلاً '30 سال، آقا، پرادراری') یا برای برسی دقیق تر واژه «سوال» را وارد کنید."
    logging.info("Forwarding miscellaneous input to Gemini API")
    gemini_response = get_gemini_response(user_input, user_id=user_id)
    return gemini_response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)