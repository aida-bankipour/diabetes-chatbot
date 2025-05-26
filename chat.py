from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import logging
import re
from datetime import datetime
import tensorflow as tf
import google.generativeai as genai
import uuid
from flask import session

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
    "پرادراری": ["پرادراری", "ادرار زیاد", "ادرار بیش از حد", "زیاد دستشویی می‌رم", "دستشویی رفتن زیاد", "شب‌ها بیدار می‌شم برای ادرار", "تکرر ادرار", "تکررادرار"],
    "عطش": ["عطش", "تشنگی", "خیلی تشنه‌ام", "مدام آب می‌خورم", "زیاد آب می‌خورم", "پرنوشی", "استسقاء"],
    "کاهش وزن": ["کاهش وزن", "افت وزن", "وزنم کم شده", "بدون دلیل وزن کم کردم"],
    "ضعف": ["ضعف", "بی‌حالی", "انرژی ندارم", "همیشه خسته‌ام", "احساس ضعف"],
    "پرخوری": ["پرخوری", "زیاد می‌خورم", "اشتهام زیاد شده", "گرسنگی مداوم دارم"],
    "عفونت قارچی": ["عفونت قارچی", "عفونت در ناحیه تناسلی", "سوزش یا خارش ناحیه تناسلی"],
    "تاری دید": ["تاری دید", "کاهش میدان دید", "چشمام تار می‌بینه", "دیدم خوب نیست"],
    "خارش": ["خارش", "خشکی پوست", "خارش بدن", "پوستم می‌خاره"],
    "عصبانیت": ["عصبانیت", "تحریک‌پذیری", "زود عصبی می‌شم", "کنترل احساسات سخت شده"],
    "تأخیر در بهبود": ["تأخیر در بهبود", "زخم‌هام دیر خوب می‌شن", "خوب نشدن زخم‌ها"],
    "فلج جزئی": ["فلج جزئی", "ضعف عضلانی", "عضلاتم ناتوان شدن", "نا توانی در حرکت"],
    "درد عضلانی": ["درد عضلانی", "کشیدگی عضلات", "بدنم درد می‌کنه"],
    "سفتی عضلات": ["سفتی عضلات", "خشکی عضلات", "عضلاتم گرفته", "گرفتگی عضلات", "درد عضلانی"],
    "ریزش مو": ["ریزش مو", "کم‌پشت شدن مو", "موهام میریزه"],
    "چاقی": ["چاقی", "اضافه وزن", "خیلی چاق شدم", "وزنم رفته بالا"],
    "قند خون بالا": [r"قند\s*(خون)?\s*(ناشتا(?:م)?)?\s*(\d+)", r"قند\s*(خون)?(?:م)?\s*بالا\s*(\d+)"]
}

# Structured question explanations
question_explanations = {
    "بیش از حد معمول ادرار": "یعنی بیشتر از حد معمول به دستشویی می‌روید، به‌خصوص شب‌ها.",
    "تشنگی مداوم": "یعنی همیشه احساس تشنگی می‌کنید و حتی با نوشیدن آب هم برطرف نمی‌شود.",
    "کاهش وزن ناگهانی": "یعنی بدون رژیم یا ورزش، وزنتان به‌سرعت کم شده است.",
    "ضعف بدنی": "یعنی احساس خستگی یا کمبود انرژی دارید، حتی بدون فعالیت زیاد.",
    "اشتها غیر عادی": "یعنی بیشتر از حد معمول احساس گرسنگی می‌کنید.",
    "عفونت‌های قارچی": "یعنی عفونت‌های مکرر، مثل خارش یا سوزش در ناحیه تناسلی.",
    "تاری دید": "یعنی اشیا را تار می‌بینید یا دیدتان واضح نیست.",
    "خشکی یا خارش پوست": "یعنی پوستتان خشک شده یا مدام می‌خارد.",
    "به سرعت عصبی شدن": "یعنی به‌راحتی و سریع عصبانی یا تحریک‌پذیر می‌شوید.",
    "بهبود کند زخم‌ها": "یعنی زخم‌ها یا جراحت‌هایتان دیرتر از معمول خوب می‌شوند.",
    "فلج جزئی": "یعنی ضعف یا کاهش توانایی حرکت در بخشی از بدن، مثل دست یا پا.",
    "کشیدگی یا درد عضلانی": "یعنی در فعالیت‌های روزمره، عضلاتتان درد می‌کند یا می‌گیرد.",
    "ریزش مو": "یعنی موهایتان بیشتر از حد معمول می‌ریزد یا کم‌پشت شده است.",
    "اضافه وزن": "یعنی وزنتان بیشتر از حد سالم برای قد و سن شماست."
}

# Symptom names in order of structured questions
symptom_names = [
    "پرادراری", "عطش", "کاهش وزن", "ضعف", "پرخوری", "عفونت قارچی", "تاری دید",
    "خارش", "عصبانیت", "تأخیر در بهبود", "فلج جزئی", "درد عضلانی", "ریزش مو", "چاقی"
]

# Keywords
positive_keywords = ["بله", "آره", "دارم", "بعضی وقتا", "گاهی", "اکثرا", "همیشه", "میکنم", "شده", "زیاد", "تا حدودی"]
negative_keywords = ["نه", "خیر", "ندارم", "نمی‌کنم", "نیستم", "اصلا", "هرگز", "کم", "به ندرت"]
goodbye_keywords = ["خداحافظ", "خدانگهدار", "بای", "بای بای", "بعدا می‌بینمت"]
thanks_keywords = ["ممنون", "ممنونم", "تشکر", "متشکرم"]
question_indicators = [
    "چیه", "چیست", "توضیح", "درباره", "چطور", "چگونه", "علائم", "علامت", "نشانه", "آیا",
    "چه", "کجا", "از کجا", "باید چی", "چند", "چقدر", "چگونه", "چرا", "کی", "کدام"
]
test_intent_keywords = [
    "تست دیابت", "دیابت دارم", "بررسی دیابت", "تشخیص دیابت", "می‌خوام تست کنم",
    "دیابت نوع", "آزمایش دیابت", "علائم دیابت"
]
explanation_indicators = ["منظور", "یعنی", "چیه", "چیست", "چرا", "چه جوریه", "توضیح بده"]

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
        "previous_symptoms": []  # برای ذخیره زمینه علائم قبلی
    }
    logging.info(f"Reset user state: {user_id}")

# Gemini API response
def get_gemini_response(user_message, context="general", user_id=None):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        if context == "symptom_explanation":
            prompt = (
                "شما یک پزشک متخصص دیابت هستید. "
                "کاربر درباره یکی از علائم دیابت سؤالی پرسیده. "
                "توضیحی کوتاه و دقیق (حداکثر دو جمله) درباره علامت از دیدگاه دیابت بدهید. "
                "سپس از کاربر بخواهید با بله یا خیر به سؤال اصلی پاسخ دهد. "
                f"سؤال کاربر: {user_message}\n"
                f"سؤال اصلی: {user_data[user_id]['questions'][user_data[user_id]['current_question_index']]}"
            )
        else:
            previous_symptoms = user_data[user_id].get("previous_symptoms", [])
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
        return "متأسفم، نمی‌توانم الان پاسخ بدم. لطفاً سن، جنسیت، علائم (مثل پرادراری) یا قند خون‌تون رو بگید ."

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
            f"قند خون {fasting_blood_sugar} میلی‌گرم در دسی‌لیتر خیلی پایینه (هیپوگلیسمی). "
            "لطفاً سریع یه منبع قندی (مثل آب‌میوه یا قرص گلوکز) بخورید و ۱۵ دقیقه بعد قند خون‌تون رو چک کنید. "
            "برای بررسی بیشتر میتونید، سن و علائم‌تون (مثل ضعف یا تشنگی) رو بگید."
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
                "بر اساس پاسخ‌هاتون، احتمال دیابت وجود داره. چند توصیه براتون دارم:<br>"
                "- لطفاً هرچه زودتر با پزشک متخصص مشورت کنید.<br>"
                "- آزمایش‌های کامل‌تر مثل قند خون ناشتا یا HbA1c انجام بدید.<br>"
                "- رژیم غذاییتون رو اصلاح کنید و قند و چربی رو کم کنید.<br>"
                "- ورزش منظم (حداقل ۳۰ دقیقه در روز) رو شروع کنید.<br>"
                "- اگه سابقه خانوادگی دیابت دارید، بیشتر مراقب باشید."
            )
        else:
            return (
                "بر اساس اطلاعات،خوشبختانه احتمال دیابت وجود ندارد یا حداقل پایین است. 😊<br>"
                "- سبک زندگی سالم رو ادامه بدید (تغذیه متعادل و ورزش).<br>"
                "- هر چند وقت یک‌بار چکاپ کنید.<br>"
                "- استرس رو مدیریت کنید و خواب کافی داشته باشید."
            )
    else:
        if probability > 50:
            return (
                "با توجه به اطلاعات شما، احتمال دیابت وجود دارد. بهتر است "
                "برای بررسی دقیق‌تر، کلمه «سوال» را وارد کنید."
            )
        else:
            return (
                "احتمال ابتلا به دیابت در شما پایین است. "
                "اما برای بررسی دقیق‌تر، میتوانید با وارد کردن کلمه «سوال» در یک تست کوچک شرکت کنید."
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
            return "خدانگهدار! خوشحال می‌شم بازم بتونم کمکتون کنم. 😊"
        return "خواهش می‌کنم! اگه سوال دیگه ای داشتید یا با علائم دیگری مواجه شدید روی کمک من حساب کنید."

    # 3. Handle structured question responses
    if current_data.get("waiting_for_questions", False):
        logging.info("Processing structured question response")
        current_question_index = current_data["current_question_index"]
        current_question = current_data["questions"][current_question_index].replace("آیا", "").strip("؟").strip()

        # Check for symptom explanation
        if any(indicator in user_input_clean for indicator in explanation_indicators):
            for key, explanation in question_explanations.items():
                if key.lower() in current_question.lower() and any(keyword in user_input_clean for keyword in [key.lower(), key.lower().replace(" ", "")]):
                    logging.info(f"Providing explanation for symptom: {key}")
                    return f"{explanation} لطفاً با بله یا خیر پاسخ بدید: {current_data['questions'][current_question_index]}"
            logging.info("Forwarding symptom explanation to Gemini API")
            gemini_response = get_gemini_response(user_input, context="symptom_explanation", user_id=user_id)
            return gemini_response

        if any(word in user_input_clean for word in positive_keywords):
            logging.info("Positive response to structured question")
            current_data["current_symptoms"].append(1)
            current_data["current_question_index"] += 1
        elif any(word in user_input_clean for word in negative_keywords):
            logging.info("Negative response to structured question")
            current_data["current_symptoms"].append(0)
            current_data["current_question_index"] += 1
        else:
            logging.info(f"Invalid response to structured question: {user_input}")
            for key, explanation in question_explanations.items():
                if key.lower() in current_question.lower():
                    return f"{explanation} لطفاً با بله یا خیر پاسخ بدید: {current_data['questions'][current_question_index]}"
            return f"لطفاً با بله یا خیر پاسخ بدید: {current_data['questions'][current_question_index]}"

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

    # Fasting blood sugar (priority over age)
    fbs_match = re.search(r'قند\s*(خون)?\s*(ناشتا(?:م)?)?\s*(\d+)', user_input_clean)
    if fbs_match:
        fbs_value = int(fbs_match.group(3))
        current_data["fasting_blood_sugar"] = fbs_value
        info_detected = True
        logging.info(f"Detected fasting blood sugar: {fbs_value}")
        if fbs_value < 70:
            responses.append(
                f"قند خون {fbs_value} میلی‌گرم در دسی‌لیتر خیلی پایینه (هیپوگلیسمی). "
                "لطفاً سریع یه منبع قندی (مثل آب‌میوه) بخورید و ۱۵ دقیقه بعد قند خون‌تون رو چک کنید."
            )
        elif fbs_value >= 100 and fbs_value < 126:
            responses.append(
                f"قند خون ناشتای {fbs_value} میلی‌گرم در دسی‌لیتر در محدوده پیش‌دیابت قرار داره. "
                "این یعنی ممکنه در معرض خطر دیابت باشید."
            )
        elif fbs_value >= 126 and "قند خون بالا" not in current_data["symptoms"]:
            current_data["symptoms"].append("قند خون بالا")
            logging.info("Added symptom: قند خون بالا")
            responses.append(f"قند خون ناشتای {fbs_value} میلی‌گرم در دسی‌لیتر بالاتر از حد نرماله.")

    # Age (only if no fasting blood sugar detected)
    if not fbs_match:
        age_match = re.search(r'(\d+)\s*سال', user_input, re.IGNORECASE)
        standalone_age_match = re.search(r'\b(\d+)\b(?!\s*%)', user_input_clean)
        if age_match:
            age = int(age_match.group(1))
            if 0 <= age <= 99:  # بررسی محدوده سنی معقول
                current_data["age"] = age
                info_detected = True
                logging.info(f"Detected age: {current_data['age']}")
        elif standalone_age_match:
            age = int(standalone_age_match.group(1))
            if 0 <= age <= 99:  # بررسی محدوده سنی معقول
                current_data["age"] = age
                info_detected = True
                logging.info(f"Detected age: {current_data['age']}")

    # Gender
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
    if symptoms_detected:
        current_data["symptoms"].extend(symptoms_detected)
        info_detected = True
        current_data["previous_symptoms"].extend(symptoms_detected)
        logging.info(f"Detected symptoms: {symptoms_detected}")

    # Check for "no symptoms"
    if "علائمی ندارم" in user_input_clean or "هیچ علامتی" in user_input_clean:
        logging.info("Detected no symptoms")
        reset_user_state(user_id)
        return "به نظر می‌رسه مشکلی ندارید! براتون آرزوی سلامتی می‌کنم. 😊 اگه علائم جدیدی پیدا کردید، بگید تا بررسی کنیم."

    # 5. Request missing information based on provided data
    if info_detected:
        missing_info = []
        if current_data["age"] is None:
            missing_info.append("سن‌تون")
        if current_data["gender"] is None:
            missing_info.append("جنسیت‌تون")
        if not current_data["symptoms"] and current_data["fasting_blood_sugar"] is None:
            missing_info.append("علائم‌تون (مثل پرادراری، تشنگی) یا قند خون ناشتا")

        if missing_info:
            logging.info(f"Requesting missing information: {', '.join(missing_info)}")
            responses.append(f"لطفاً {', '.join(missing_info)} رو بگید تا بتونم بررسی دقیق‌تری انجام بدم.")
            return ", ".join(responses)

    # 6. Handle test intent or structured questions request
    test_intent = any(keyword in user_input_clean for keyword in test_intent_keywords)
    if user_input_clean in ["سوال", "بپرس", "پرسش", "باشه", "شروع کن"] or test_intent:
        if current_data["age"] is not None and current_data["gender"] is not None and (current_data["symptoms"] or current_data["fasting_blood_sugar"] is not None):
            current_data["waiting_for_questions"] = True
            current_data["current_question_index"] = 0
            current_data["current_symptoms"] = []
            logging.info("Starting structured questions")
            return current_data["questions"][0]
        else:
            logging.info("Insufficient data for structured questions")
            missing_info = []
            if current_data["age"] is None:
                missing_info.append("سن‌تون")
            if current_data["gender"] is None:
                missing_info.append("جنسیت‌تون (آقا یا خانم)")
            if not current_data["symptoms"] and current_data["fasting_blood_sugar"] is None:
                missing_info.append("علائم‌تون (مثل پرادراری، تشنگی) یا قند خون ناشتا")
            responses.append(f"لطفاً {', '.join(missing_info)} رو بگید تا تست رو شروع کنیم.")
            return ", ".join(responses)

    # 7. Perform prediction if all data provided
    if (current_data["age"] is not None and
        current_data["gender"] is not None and
        (current_data["symptoms"] or current_data["fasting_blood_sugar"] is not None)):
        if not current_data["prediction_done"]:
            logging.info("Performing initial prediction")
            current_data["prediction_done"] = True
            prediction_result = predict_diabetes_response(current_data)
            responses.append(f"ممنون که گفتید سن‌تون {current_data['age']} ساله و جنسیت‌تون {'خانم' if current_data['gender'] == 0 else 'آقا'} است. {prediction_result}")
            return ", ".join(responses)

    # 8. Handle irrelevant symptoms or general questions
    if info_detected and not current_data["symptoms"] and current_data["fasting_blood_sugar"] is None:
        logging.info("Forwarding irrelevant symptoms to Gemini API")
        gemini_response = get_gemini_response(user_input, user_id=user_id)
        responses.append(gemini_response)
        return ", ".join(responses)

    # 9. Handle general questions or miscellaneous input
    if any(indicator in user_input_clean for indicator in question_indicators):
        logging.info("Detected general question")
        gemini_response = get_gemini_response(user_input, user_id=user_id)
        responses.append(gemini_response)
        return ", ".join(responses)

    # 10. Handle greetings or unknown input
    if user_input_clean in ["سلام", "سلام علکیم", "سلام خوبی"]:
        return "سلام! 😊 برای بررسی دیابت، لطفاً سن، جنسیت، علائم (مثل پرادراری) یا قند خون‌تون رو بگید (مثلاً '30 سال، آقا، پرادراری') یا بنویسید «سوال» برای برسی دقیق تر."
    logging.info("Forwarding miscellaneous input to Gemini API")
    gemini_response = get_gemini_response(user_input, user_id=user_id)
    return gemini_response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


 
