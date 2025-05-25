from flask import Flask, render_template, request, jsonify
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
    "پرادراری": ["پرادراری", "ادرار زیاد", "ادرار بیش از حد", "زیاد دستشویی می‌رم", "دستشویی رفتن زیاد", "شب‌ها بیدار می‌شم برای ادرار"],
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
    "قند خون بالا": ["قند خون \d+", "قند بالا \d+", "قند خون ناشتا \d+"]
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
        "fasting_blood_sugar": 0,
        "waiting_for_questions": False,
        "current_question_index": 0,
        "current_symptoms": [],
        "prediction_done": False
    }
    logging.info(f"Reset user state: {user_id}")

# Gemini API response
def get_gemini_response(user_message, context="general"):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        if context == "symptom_explanation":
            prompt = (
                "شما یک پزشک متخصص دیابت هستید. "
                "کاربر درباره یکی از علائم دیابت (پرادراری، عطش، کاهش وزن، ضعف، پرخوری، عفونت قارچی، تاری دید، خارش، عصبانیت، تأخیر در بهبود، فلج جزئی، درد عضلانی، ریزش مو، چاقی) سؤالی پرسیده. "
                "توضیحی کوتاه و دقیق درباره علامت از دیدگاه دیابت بدهید و از کاربر بخواهید با بله یا خیر به سؤال اصلی پاسخ دهد. "
                "پاسخ را به زبان فارسی و حداکثر در دو جمله ارائه کنید. "
                f"سؤال کاربر: {user_message}\n"
                f"سؤال اصلی: {questions[current_question_index]}"
            )
        else:
            prompt = (
                "شما یک پزشک عمومی و متخصص دیابت هستید. "
                "به سؤالم پاسخ دهید و پاسخ را کوتاه، دقیق و به زبان فارسی ارائه کنید. "
                "فقط اطلاعاتی مرتبط با دیابت یا سلامت عمومی ارائه دهید. "
                "اگر سؤال نامفهوم یا نامرتبط است، کاربر را به بررسی علائم دیابت هدایت کنید. "
                f"سؤال کاربر: {user_message}"
            )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return "متأسفم، نمی‌توانم الان پاسخ بدم. لطفاً دوباره امتحان کنید."

# Predict diabetes probability
def predict_diabetes(input_data):
    try:
        prediction = model.predict(input_data, verbose=0)
        probability = prediction[0][0] * 100
        return probability
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return 0

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["message"].strip()
    user_id = request.form.get("user_id", str(uuid.uuid4()))
    
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
    global questions, current_question_index
    current_data = user_data[user_id]
    user_input_clean = user_input.lower().replace("‌", "")
    responses = []

    # Structured questions
    questions = [
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
    ]

    # 1. Check for goodbye
    if any(word == user_input_clean for word in goodbye_keywords):
        logging.info("Detected goodbye")
        reset_user_state(user_id)
        return "خدانگهدار! امیدوارم تونسته باشم کمکتون کنم.! 😊"

    # 2. Check for thanks
    if any(word in user_input_clean for word in thanks_keywords):
        logging.info("Detected thanks")
        return "خواهش می‌کنم!آیا میتونم کمک دیگه ای به شما بکنم؟ 🌷"

    # 3. Check for structured questions request
    if user_input_clean in ["سوال", "بپرس", "پرسش", "باشه", "شروع کن"]:
        if current_data["symptoms"] or current_data["fasting_blood_sugar"] >= 126:
            current_data["waiting_for_questions"] = True
            current_data["current_question_index"] = 0
            current_data["current_symptoms"] = []
            logging.info("Starting structured questions")
            return questions[0]
        else:
            logging.info("No symptoms for structured questions")
            return "لطفاً اول علائم خود رو بگید تا سؤالات دقیق‌تری بپرسم!"

    # 4. Handle structured question responses
    if current_data.get("waiting_for_questions", False):
        logging.info("Processing structured question response")
        current_question_index = current_data["current_question_index"]
        current_question = questions[current_question_index].replace("آیا", "").strip("؟").strip()

        # Check for symptom explanation
        if any(indicator in user_input_clean for indicator in explanation_indicators):
            for key, explanation in question_explanations.items():
                if key.lower() in current_question.lower() and any(keyword in user_input_clean for keyword in [key.lower(), key.lower().replace(" ", "")]):
                    logging.info(f"Providing explanation for symptom: {key}")
                    return f"{explanation} لطفاً به سؤالم با بله یا خیر پاسخ بدید: {questions[current_question_index]}"
            # If no exact match, forward to Gemini API
            logging.info("Forwarding symptom explanation to Gemini API")
            gemini_response = get_gemini_response(user_input, context="symptom_explanation")
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
            return f"لطفاً به سؤالم با بله یا خیر پاسخ بدید: {questions[current_question_index]}"

        if current_data["current_question_index"] < len(questions):
            return questions[current_data["current_question_index"]]
        else:
            logging.info("Finished structured questions")
            logging.info(f"Current symptoms: {current_data['current_symptoms']}")
            current_data["waiting_for_questions"] = False
            prediction_result = predict_diabetes_response(current_data, detailed=True)
            probability = predict_diabetes(
                np.array([[current_data["age"], current_data["gender"]] + current_data["current_symptoms"]])
            )
            current_data["symptoms"] = [
                symptom_names[i] for i, val in enumerate(current_data["current_symptoms"]) if val == 1
            ]
            reset_user_state(user_id)
            return prediction_result

    # 5. Extract information
    logging.info("Extracting information")
    info_detected = False

    # Age
    age_match = re.search(r'(\d+)\s*سال', user_input, re.IGNORECASE)
    standalone_age_match = re.search(r'^\d+$', user_input_clean)
    if age_match:
        current_data["age"] = int(age_match.group(1))
        info_detected = True
        logging.info(f"Detected age: {current_data['age']}")
    elif standalone_age_match:
        current_data["age"] = int(standalone_age_match.group(0))
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

    # Fasting blood sugar
    fbs_match = re.search(r'قند\s*(خون)?\s*(ناشتا)?\s*(\d+)', user_input_clean)
    if fbs_match:
        fbs_value = int(fbs_match.group(3))
        current_data["fasting_blood_sugar"] = fbs_value
        info_detected = True
        logging.info(f"Detected fasting blood sugar: {fbs_value}")
        responses.append(f"قند خون ناشتای {fbs_value} میلی‌گرم در دسی‌لیتر {'در محدوده نرمال است' if fbs_value < 126 else 'بالاتر از حد نرمال است'}.")
        if fbs_value >= 126 and "قند خون بالا" not in current_data["symptoms"]:
            current_data["symptoms"].append("قند خون بالا")
            logging.info("Added symptom: قند خون بالا")

    # Symptoms
    symptoms_detected = []
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
        logging.info(f"Detected symptoms: {symptoms_detected}")

    # Reset state for new conversation
    if info_detected and all(v is None or v == 0 or v == [] for k, v in current_data.items() if k not in ["current_question_index", "current_symptoms", "prediction_done"]):
        logging.info("Detected new conversation, resetting state")
        reset_user_state(user_id)
        current_data = user_data[user_id]
        if age_match:
            current_data["age"] = int(age_match.group(1))
        elif standalone_age_match:
            current_data["age"] = int(standalone_age_match.group(0))
        if any(g in user_input_clean for g in ["خانم", "زن", "دختر", "مونث"]):
            current_data["gender"] = 0
        elif any(g in user_input_clean for g in ["آقا", "مرد", "پسر", "مذکر"]):
            current_data["gender"] = 1
        if fbs_match:
            current_data["fasting_blood_sugar"] = fbs_value
            if fbs_value >= 126:
                current_data["symptoms"].append("قند خون بالا")
        current_data["symptoms"].extend(symptoms_detected)

    # 6. Check for general questions or test intent
    pure_info = (
        re.match(r'^\d+$', user_input_clean) or
        re.match(r'^(خانم|زن|دختر|مونث|آقا|مرد|پسر|مذکر)$', user_input_clean) or
        (symptoms_detected and not any(indicator in user_input_clean for indicator in question_indicators))
    )
    test_intent = any(keyword in user_input_clean for keyword in test_intent_keywords)
    if (any(indicator in user_input_clean for indicator in question_indicators) or not info_detected) and not pure_info:
        logging.info("Detected general question, forwarding to Gemini API")
        if info_detected and current_data["age"] is not None and current_data["gender"] is not None and (current_data["symptoms"] or current_data["fasting_blood_sugar"] >= 126):
            if not current_data["prediction_done"]:
                logging.info("Performing initial prediction")
                current_data["prediction_done"] = True
                responses.append(predict_diabetes_response(current_data))
        gemini_response = get_gemini_response(user_input)
        responses.append(gemini_response)
        return ", ".join(responses)
    elif test_intent:
        logging.info("Detected test intent")
        return "باشه! برای بررسی دیابت، لطفاً سن، جنسیت و علائمی که دارید و بگید."

    # 7. Request missing information
    if info_detected:
        # Perform prediction if enough data
        if (current_data["age"] is not None and
            current_data["gender"] is not None and
            (current_data["symptoms"] or current_data["fasting_blood_sugar"] >= 126)):
            if not current_data["prediction_done"]:
                logging.info("Performing initial prediction")
                current_data["prediction_done"] = True
                prediction_result = predict_diabetes_response(current_data)
                responses.append(prediction_result)
                return ", ".join(responses)

        # Request missing information
        if current_data["age"] is None:
            logging.info("Requesting age")
            responses.append("لطفاً سن خودتون رو بگید.")
            return ", ".join(responses)
        elif current_data["gender"] is None:
            logging.info("Requesting gender")
            responses.append("لطفاً جنسیت خودتون و مشخص کنید (آقا یا خانم).")
            return ", ".join(responses)
        elif not current_data["symptoms"] and current_data["fasting_blood_sugar"] < 126:
            logging.info("Requesting symptoms")
            responses.append("لطفاً علائمتان را بگویید (مثلاً پرادراری، تشنگی، ضعف یا ...).")
            return ", ".join(responses)

    # 8. Handle miscellaneous input
    logging.info(f"Miscellaneous input: {user_input}")
    if user_input_clean in ["سلام", "سلام علکیم", "سلام خوبی"]:
        return "سلام! 😊 چطور می‌تونم بهتون کمک کنم؟ اگه می‌خواید دیابت رو بررسی کنیم، سن، جنسیت یا علائمتون رو بگید."
    # Forward unknown input to Gemini API
    logging.info("Forwarding miscellaneous input to Gemini API")
    gemini_response = get_gemini_response(user_input)
    return gemini_response

# Diabetes prediction response
def predict_diabetes_response(data, detailed=False):
    age = data["age"]
    gender = data["gender"]
    if detailed:
        symptoms = data["current_symptoms"]
    else:
        # Map textual symptoms to binary features
        symptoms = [0] * len(symptom_names)
        for symptom in data["symptoms"]:
            if symptom in symptom_names:
                symptoms[symptom_names.index(symptom)] = 1

    input_features = np.array([[
        age,
        gender,
        symptoms[0] if len(symptoms) > 0 else 0,  # پرادراری
        symptoms[1] if len(symptoms) > 1 else 0,  # عطش
        symptoms[2] if len(symptoms) > 2 else 0,  # کاهش وزن
        symptoms[3] if len(symptoms) > 3 else 0,  # ضعف
        symptoms[4] if len(symptoms) > 4 else 0,  # پرخوری
        symptoms[5] if len(symptoms) > 5 else 0,  # عفونت قارچی
        symptoms[6] if len(symptoms) > 6 else 0,  # تاری دید
        symptoms[7] if len(symptoms) > 7 else 0,  # خارش
        symptoms[8] if len(symptoms) > 8 else 0,  # عصبانیت
        symptoms[9] if len(symptoms) > 9 else 0,  # تأخیر در بهبود
        symptoms[10] if len(symptoms) > 10 else 0,  # فلج جزئی
        symptoms[11] if len(symptoms) > 11 else 0,  # درد عضلانی
        symptoms[12] if len(symptoms) > 12 else 0,  # ریزش مو
        symptoms[13] if len(symptoms) > 13 else 0   # چاقی
    ]], dtype=float)

    logging.info(f"Input features: {input_features}")
    probability = predict_diabetes(input_features)
    logging.info(f"Prediction probability: {probability}")

    if data["fasting_blood_sugar"] >= 126 or "قند خون بالا" in data["symptoms"]:
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
                "بر اساس اطلاعات، احتمال دیابت خیلی کمه. 😊<br>"
                "- سبک زندگی سالم رو ادامه بدید (تغذیه متعادل و ورزش).<br>"
                "- هر چند وقت یک‌بار چکاپ کنید.<br>"
                "- استرس رو مدیریت کنید و خواب کافی داشته باشید."
            )
    else:
        if probability > 50:
            return (
                "احتمال ابتلا به دیابت وجود داره. می‌تونم با چند سؤال دقیق‌تر بررسی کنم. "
                "اگه موافق برسی دقیق تر هستید عبارت «سوال» را وارد کنید."
            )
        else:
            return (
                "احتمال ابتلا به دیابت پایینه. اما میتونیم تست دقیق تری داشته باشیم اگر موافقید واژه «سوال» بنویسد تا فرآیند برسی دقیق تر شروع کنیم."
            )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


 
