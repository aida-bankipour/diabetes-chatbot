from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import tensorflow as tf
import logging

# تنظیمات اولیه logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# بارگذاری مدل
model_path = 'mlp_model/mlp_model.keras'
model = tf.keras.models.load_model(model_path)

# دیکشنری برای نگهداری وضعیت کاربر
user_data = {}

# لیست علائم معتبر
valid_symptoms = [
    "پرادراری", "ادرار بیش از حد معمول","ادرار زیاد",
    "عطش","تشنگی", "کاهش وزن","کاهش ناگهانی وزن",
    "افت وزن","افت شدید وزن","ضعف","بی حالی", "پرخوری",
    "عصبانیت","عصبی","عفونت قارچی", "تاری دید","کاهش میدان دید",
    "خارش", "خشکی", "تحریک‌پذیری","تأخیر در بهبود", "فلج جزئی",
    "درد عضلانی","کشیدگی", "سفتی عضلات", "ریزش مو", "چاقی", "درد"
]

def predict_diabetes(input_data):
    prediction = model.predict(input_data)
    probability = prediction[0][0] * 100
    return probability

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["message"]
    user_id = request.form.get("user_id")
   
    if user_id not in user_data:
        user_data[user_id] = {
            "age": None,
            "gender": None,
            "symptoms": []
        }
    
    logging.info(f"ورودی کاربر: {user_message}")
    response = user_response(user_message, user_id)
    return jsonify({"response": response})

def user_response(user_input, user_id):
    current_data = user_data[user_id]
    responses = []
    questions = [
        "آیا بیش از حد معمول ادرار می‌کنید؟",
        "آیا احساس تشنگی مداوم دارید؟",
        "آیا کاهش وزن ناگهانی داشته‌اید؟",
        "آیا ضعف بدنی دارید؟",
        "آیا اشتها شما به طور غیر عادی افزایش پیداکرده است؟",
        "آیا مبتلا به عفونت‌های قارچی هستید؟",
        "آیا تاری دید دارید؟",
        "آیا احساس خشکی و یا خارش پوست دارید؟",
        "آیا به سرعت عصبی می‌شوید؟",
        "آیا بهبود زخم‌های بدنتان به کندی صورت می‌گیرد؟",
        "آیا فلج جزئی دارید؟",
        "آیا در عضله خاصی احساس کشیدگی یا درد می‌کنید؟",
        "آیا ریزش مو دارید؟",
        "آیا اضافه وزن دارید؟"
    ]

    if "سلام" in user_input.strip():
        return "سلام به چت بات تشخیص دیابت خوش آمدید!<br>چطور می‌توانم به شما کمک کنم؟"
    if "دیابت" in user_input.strip():
        return "لطفاً سن و جنسیت و علائم خود را وارد کنید.<br> تا احتمال ابتلا شما به دیابت را بررسی کنم."
    elif current_data.get("waiting_for_more_questions", False):
        if user_input.strip() in ["سوال", "بپرس", "پرسش", "باشه"]:
            current_data["current_question_index"] = 0
            return questions[current_data["current_question_index"]]

    if user_input.strip() in ["خداحافظ", "خدافظ", "خدانگهدار", "بای"]:
        return "امیدوارم توانسته باشم به شما کمک کنم.<br> خدانگهدار."

    if current_data["gender"] is None:
        if any(word in user_input for word in ["خانم", "زن", "دختر"]):
            current_data["gender"] = 0
        elif any(word in user_input for word in ["آقا", "مرد", "پسر"]):
            current_data["gender"] = 1

    age_match = re.search(r'(\d+)\s*سال', user_input)
    if age_match and current_data["age"] is None:
        current_data["age"] = int(age_match.group(1))
    elif current_data["age"] is None:
        standalone_age_match = re.search(r'^\d+$', user_input.strip())
        if standalone_age_match:
            current_data["age"] = int(standalone_age_match.group(0))
        else:
            return "لطفاً سن خود را وارد کنید."

    symptoms_in_input = []
    for symptom in valid_symptoms:
        if symptom in user_input and symptom not in current_data["symptoms"]:
            symptoms_in_input.append(symptom)

    if symptoms_in_input:
        current_data["symptoms"].extend(symptoms_in_input)

    if current_data["age"] is not None and current_data["gender"] is not None:
        if not current_data.get("prediction_done", False):
            current_data["prediction_done"] = True
            prediction_result = predict_diabetes_response(current_data)
            responses.append(prediction_result)
            current_data["waiting_for_more_questions"] = True
            current_data["current_question_index"] = 0
            current_data["current_symptoms"] = []
            return responses

        else:
            if current_data.get("waiting_for_more_questions", False):
                positive_answers = ["بله", "دارم", "اره", "زیاد", "تا حدودی", "بیشتر مواقع", "اکثرا", "هستم", "میکنم"]
                negative_answers = ["ندارم", "خیر", "نه", "کم", "نداشتم", "اصلا", "نمیکنم", "نیستم"]

                if user_input.lower() in positive_answers:
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() in negative_answers:
                    current_data["current_symptoms"].append(0)

                current_data["current_question_index"] += 1

                if current_data["current_question_index"] < len(questions):
                    return questions[current_data["current_question_index"]]
                else:
                    responses.append("سوالات به پایان رسید.<br> پیش‌بینی دقیق‌تری انجام می‌دهم....")
                    current_data["symptoms"] = current_data["current_symptoms"]
                    final_prediction_result = predict_more_accurate_diabetes_response(current_data)
                    responses.append(final_prediction_result)
                    current_data["waiting_for_more_questions"] = False

    if current_data["age"] is None:
        return "لطفاً سن خود را وارد کنید."
    if current_data["gender"] is None:
        return "لطفاً جنسیت خود را وارد کنید (خانم یا آقا)."

    return " ".join(responses)

def predict_diabetes_response(data):
    age = data["age"]
    gender = data["gender"]

    polyuria = 1 if any(x in data["symptoms"] for x in ['ادرار بیش از حد معمول', 'پرادراری', 'ادرار زیاد']) else 0
    polydipsia = 1 if any(x in data["symptoms"] for x in ['عطش', 'تشنگی']) else 0
    sudden_weight_loss = 1 if any(x in data["symptoms"] for x in ['کاهش وزن', 'کاهش ناگهانی وزن', 'افت وزن', 'افت شدید وزن']) else 0
    weakness = 1 if any(x in data["symptoms"] for x in ['ضعف', 'بی حالی']) else 0
    polyphagia = 1 if 'پرخوری' in data["symptoms"] else 0
    genital_thrush = 1 if 'عفونت قارچی' in data["symptoms"] else 0
    visual_blurring = 1 if any(x in data["symptoms"] for x in ['تاری دید', 'کاهش میدان دید']) else 0
    itching = 1 if any(x in data["symptoms"] for x in ['خارش', 'خشکی']) else 0
    irritability = 1 if any(x in data["symptoms"] for x in ['تحریک‌ پذیری', 'عصبانیت', 'عصبی']) else 0
    delayed_healing = 1 if 'تأخیر در بهبود' in data["symptoms"] else 0
    partial_paresis = 1 if any(x in data["symptoms"] for x in ['فلج جزئی', 'درد عضلانی', 'کشیدگی']) else 0
    muscle_stiffness = 1 if any(x in data["symptoms"] for x in ['سفتی عضلات', 'درد']) else 0
    alopecia = 1 if 'ریزش مو' in data["symptoms"] else 0
    obesity = 1 if 'چاقی' in data["symptoms"] else 0

    input_features = np.array([[age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia,
                                genital_thrush, visual_blurring, itching, irritability, delayed_healing,
                                partial_paresis, muscle_stiffness, alopecia, obesity]])

    logging.info(f"ویژگی‌های ورودی برای مدل: {input_features}")
    probability = predict_diabetes(input_features)
    logging.info(f"احتمال پیش‌بینی: {probability}")

    if probability > 50:
        return ("🔮پیش‌بینی: بر اساس علائمی که وارد کردید احتمال ابتلا به دیابت وجود دارد.<br>آیا مایل هستید چند سوال بپرسم و پیش‌بینی دقیق‌تری انجام دهم؟ (اگر مایل هستید کلمه سوال را وارد کنید)")
    else:
        return ("🔮پیش‌بینی: بر اساس علائمی که وارد کردید احتمال ابتلا به دیابت وجود ندارد.<br>آیا مایل هستید چند سوال بپرسم و پیش‌بینی دقیق‌تری انجام دهم؟ (اگر مایل هستید کلمه سوال را وارد کنید)")

if __name__ == "__main__":
    app.run(debug=False)

 
