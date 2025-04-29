from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import csv
import logging
import re
from datetime import datetime
import tensorflow as tf 
import logging 

# Set up logging
logging.basicConfig(level=logging.DEBUG)
# تنظیمات اولیه logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)


# بارگذاری مدل  
# بارگذاری مدل
model_path = 'mlp_model/mlp_model.keras'
model = tf.keras.models.load_model(model_path) 

# Dictionary to keep track of user state
user_data = {}

# List of valid symptoms
symptom_keywords = {
    "پرادراری": ["پرادراری", "ادرار زیاد", "ادرار بیش از حد", "زیاد دستشویی می‌رم", "دستشویی رفتن زیاد", "شب‌ها بیدار می‌شم برای ادرار", "شب‌ها زیاد بیدار می‌شم"],
    "عطش": ["عطش", "تشنگی", "خیلی تشنه‌ام", "مدام آب می‌خورم", "زیاد آب می‌خورم"],
    "کاهش وزن": ["کاهش وزن", "افت وزن", "وزنم کم شده", "بدون دلیل وزن کم کردم", "ناگهانی وزن کم کردم"],
    "ضعف": ["ضعف", "بی‌حالی", "انرژی ندارم", "همیشه خسته‌ام", "خیلی بی‌حال شدم"],
    "پرخوری": ["پرخوری", "زیاد می‌خورم", "اشتهام زیاد شده", "گرسنگی مداوم دارم"],
    "عفونت قارچی": ["عفونت قارچی", "عفونت در ناحیه تناسلی", "سوزش یا خارش ناحیه تناسلی", "بوی نامطبوع"],
    "تاری دید": ["تاری دید", "کاهش میدان دید", "چشمام تار می‌بینه", "دیدم خوب نیست", "چیزا رو تار می‌بینم"],
    "خارش": ["خارش", "خشکی پوست", "خارش بدن", "پوستم می‌خاره", "پوستم خشک شده"],
    "عصبانیت": ["عصبانیت", "تحریک‌پذیری", "زود از کوره در می‌رم", "زود عصبی می‌شم", "کنترل احساسات سخت شده"],
    "تأخیر در بهبود": ["تأخیر در بهبود", "زخم‌هام دیر خوب می‌شن", "زخم‌هام باقی می‌مونن", "خوب نشدن زخم‌ها"],
    "فلج جزئی": ["فلج جزئی", "ضعف عضلانی", "عضلاتم ناتوان شدن", "بخشی از بدنم خوب کار نمی‌کنه"],
    "درد عضلانی": ["درد عضلانی", "کشیدگی عضلات", "بدنم درد می‌کنه", "عضلاتم می‌کشه"],
    "سفتی عضلات": ["سفتی عضلات", "خشکی عضلات", "عضلاتم گرفته", "انعطاف ندارم"],
    "ریزش مو": ["ریزش مو", "کم‌پشت شدن مو", "موهام میریزه"],
    "چاقی": ["چاقی", "اضافه وزن", "خیلی چاق شدم", "وزنم رفته بالا"]
}

positive_keywords = [
    "بله", "آره", "اره", "دارم", "بعضی وقتا", "گاهی", "اکثرا", "همیشه", 
    "میکنم", "احساس میکنم", "شده", "پیش میاد", "زیاد", "تا حدودی", "درگیرم",
    "برام پیش اومده", "مشاهده کردم", "دیدم", "می‌ شوم", "احساس می‌کنم", "دچارم", "هستم"
]

negative_keywords = [
    "نه", "خیر", "ندارم", "نمی‌کنم", "نیستم", "نمی‌شوم", "نمی‌خورم", "نمی‌رم", 
    "اصلا", "نداشتم", "هرگز", "کم", "خیلی کم", "نادره", "به ندرت", "تقریباً نه"
]

# FAQ responses in Persian
faq_responses = {
    "علائم دیابت": 
    "برخی علائم دیابت عبارتند از: پرادراری، تشنگی زیاد، کاهش وزن ناگهانی",
    
    "دیابت چیست": 
    "دیابت یک بیماری مزمن است که در آن سطح قند خون (گلوکز) در بدن افزایش می‌یابد",

    "علائم دیابت چیست": 
    "برخی علائم دیابت عبارتند از: پرادراری، تشنگی زیاد، کاهش وزن ناگهانی",

    "چگونه از دیابت پیشگیری کنیم": 
    "با تغذیه سالم، ورزش منظم، کنترل وزن و چکاپ‌های منظم",

    "علائم دیابت": 
    "برخی علائم دیابت عبارتند از: پرادراری، تشنگی زیاد، کاهش وزن ناگهانی",

    "پیشگیری از دیابت":
    "پیشگیری از دیابت نیازمند تغذیه سالم و ورزش منظم است . همچنین چکاپ های منظم و کنترل وزن نیز میتواند به پیشگیری از دیابت کمک کند.",

    "درمان دیابت چگونه است": 
    "درمان دیابت شامل رژیم غذایی مناسب، ورزش، مصرف دارو یا انسولین",

    "چه علائمی نشان‌دهنده دیابت هستند" : 
    "پرادراری، تشنگی زیاد، کاهش وزن، خستگی، تاری دید، دیر خوب شدن زخم‌ها و عفونت‌های مکرر از جمله علائم دیابت هستند.",

    "دیابت نوع ۱ چیست" : 
    "دیابت نوع ۱ یک بیماری خودایمنی است که در آن بدن سلول‌های تولیدکننده انسولین در پانکراس را از بین می‌برد و معمولاً در کودکان و نوجوانان تشخیص داده می‌شود.",

    "دیابت نوع یک چیست" : 
    "دیابت نوع ۱ یک بیماری خودایمنی است که در آن بدن سلول‌های تولیدکننده انسولین در پانکراس را از بین می‌برد و معمولاً در کودکان و نوجوانان تشخیص داده می‌شود.",

    "دیابت نوع ۲ چیست" :
    " دیابت نوع ۲ شایع‌ترین نوع دیابت است که معمولاً در افراد بزرگ‌سال بروز می‌کند و ناشی از مقاومت بدن به انسولین یا کاهش تولید آن است.",

    "دیابت نوع دو چیست" : 
    "دیابت نوع ۲ شایع‌ترین نوع دیابت است که معمولاً در افراد بزرگ‌سال بروز می‌کند و ناشی از مقاومت بدن به انسولین یا کاهش تولید آن است.",

    "آیا دیابت درمان دارد" :
    "دیابت درمان قطعی ندارد، اما با تغییر سبک زندگی، رژیم غذایی مناسب، دارو و انسولین می‌توان آن را کنترل کرد",

    "آیا دیابت ارثی است" :
    "بله، سابقه خانوادگی یکی از عوامل خطر ابتلا به دیابت است، اما عوامل محیطی نیز در بروز آن نقش دارند.",

    "چگونه می‌توان از دیابت پیشگیری کرد" :
    "با تغذیه سالم، ورزش منظم، حفظ وزن مناسب و چکاپ‌های دوره‌ای می‌توان از بروز دیابت نوع ۲ جلوگیری کرد.",

    "نقش انسولین در بدن چیست" :
    "انسولین هورمونی است که به انتقال گلوکز از خون به سلول‌ها کمک می‌کند تا از آن به عنوان انرژی استفاده کنند.",

    "اگر دیابت کنترل نشود، چه عوارضی دارد" :
    "عوارض دیابت شامل بیماری‌های قلبی، آسیب به کلیه، مشکلات چشمی، آسیب عصبی و زخم‌های مزمن است.",

    "دیابت بارداری چیست" :
    "دیابت بارداری نوعی دیابت است که برای اولین بار در دوران بارداری تشخیص داده می‌شود و معمولاً پس از زایمان برطرف می‌شود.",

    "چگونه قند خون را کنترل کنیم":
    "با رعایت رژیم غذایی سالم، فعالیت بدنی منظم، مصرف دارو یا انسولین و اندازه‌گیری منظم قند خون می‌توان آن را کنترل کرد.",

    "آیا افراد لاغر هم دیابت می‌گیرند" :
    "بله، گرچه چاقی یکی از عوامل خطر است، اما دیابت می‌تواند در افراد لاغر نیز به دلایل ژنتیکی یا دیگر عوامل رخ دهد.",

    "آیا دیابت باعث افسردگی می‌شود" :
    "افراد مبتلا به دیابت به دلیل فشار روانی بیماری بیشتر در معرض افسردگی قرار دارند و نیاز به حمایت روحی دارند.",

    "چه غذاهایی برای دیابتی‌ها مناسب هستند" :
    "غذاهای کم‌قند، کم‌چرب، دارای فیبر بالا مانند سبزیجات، غلات کامل، گوشت بدون چربی و لبنیات کم‌چرب مناسب هستند.",

    "آیا می‌توان از انسولین طبیعی استفاده کرد" :
    "در حال حاضر انسولین موجود در بازار به صورت دارویی و تولید شده در آزمایشگاه‌ها است و انسولین طبیعی برای درمان استفاده نمی‌شود.",

    "آیا دیابت قابل بازگشت است" :
    "در برخی موارد دیابت نوع ۲ با کاهش وزن و تغییر سبک زندگی ممکن است بهبود یابد، اما به معنی درمان قطعی نیست.",

    "چگونه از زخم پای دیابتی جلوگیری کنیم" :
    "بررسی روزانه پاها، پوشیدن کفش مناسب، رعایت بهداشت و کنترل قند خون از راه‌های پیشگیری از زخم پای دیابتی است.",

    "آیا استرس روی دیابت تاثیر دارد":
    "بله، استرس می‌تواند باعث افزایش سطح قند خون شود و کنترل دیابت را سخت‌تر کند.",

    "چه آزمایش‌هایی برای تشخیص دیابت انجام می‌شود" :
    "آزمایش قند خون ناشتا، آزمایش HbA1c و تست تحمل گلوکز برای تشخیص دیابت استفاده می‌شوند.",

    "آیا دیابت قابل کنترل است" :
    "بله، دیابت با پایش منظم، رعایت رژیم غذایی، ورزش و مصرف داروها قابل کنترل است و افراد می‌توانند زندگی سالمی داشته باشند."
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])  
def get_response():  
    user_message = request.form["message"]  
    user_id = request.form.get("user_id")  # گرفتن شناسه کاربر برای نگهداری وضعیت  

    # ایجاد ساختار برای ذخیره اطلاعات کاربر اگر هنوز وجود نداشته باشد  
    if user_id not in user_data:  
        user_data[user_id] = {  
            "age": None,  
            "gender": None,  
            "symptoms": []  
        }  

    print(f"ورودی کاربر: {user_message}")  
    response = user_response(user_message, user_id)  
    return jsonify({"response": response})

# Prediction function
def predict_diabetes(input_data):
    prediction = model.predict(input_data)  
    probability = prediction[0][0] * 100  
    return probability
     
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

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
        "آیا فلج جزئی (ضعف یا کاهش توانایی حرکتی در یک عضله خاص) دارید؟",  
        "آیا در انجام فعالیت های روزمره در عضله خاصی احساس کشیدگی یا درد می‌کنید؟",  
        "آیا ریزش مو دارید؟",  
        "آیا اضافه وزن دارید؟"  
    ]

    user_input_clean = user_input.strip().lower() 

    if user_input_clean in ["خداحافظ", "خدانگهدار", "بای", "خدافظ"]:  
        return "امیدوارم تونسته باشم کمکتون کنم. خدانگهدار " 
    
    if user_input_clean in ["ممنون", "ممنونم", "تشکر", "متشکرم"]:  
        return "خواهش میکنم. امیدوارم این اطلاعات برای شما مفید بوده باشد. آیا کمک دیگری از من بر می آید؟🌷" 

    elif current_data.get("waiting_for_more_questions", False):  
        if user_input.strip() in ["سوال", "بپرس", "پرسش", "باشه", "حتما", "شروع کن", "بپرس"]:  
            current_data["current_question_index"] = 0  # شروع از سوال اول  
            return questions[current_data["current_question_index"]]


    # بررسی سوالات متداول
    cleaned_input = user_input.strip().replace("؟", "").replace("?", "").lower()
    for faq_question in faq_responses:
        if faq_question in cleaned_input:
            return faq_responses[faq_question]



    # جنسیت
    if current_data["gender"] is None:
        if any(g in user_input_clean for g in ["خانم", "زن", "دختر", "مونث"]):
            current_data["gender"] = 0
        elif any(g in user_input_clean for g in ["آقا", "مرد", "پسر", "مذکر"]):
            current_data["gender"] = 1


    # سن
    age_match = re.search(r'(\d+)\s*سال', user_input)
    if age_match and current_data["age"] is None:
        current_data["age"] = int(age_match.group(1))
    elif current_data["age"] is None:
        standalone_age_match = re.search(r'^\d+$', user_input_clean)
        if standalone_age_match:
            current_data["age"] = int(standalone_age_match.group(0))
        else:
            return "لطفاً سن خود را وارد کنید." 

    # بررسی علائم از طریق عبارت‌های متنوع


    symptoms_in_input = []
    for symptom, keywords in symptom_keywords.items():
        for keyword in keywords:
            pattern = re.escape(keyword)
            if re.search(pattern, user_input_clean) and symptom not in current_data["symptoms"]:
                symptoms_in_input.append(symptom)
                break

    if symptoms_in_input:
        current_data["symptoms"].extend(symptoms_in_input)

    # پیش‌بینی اولیه
    if current_data["age"] is not None and current_data["gender"] is not None:
        if not current_data.get("prediction_done", False):
            current_data["prediction_done"] = True
            prediction_result = predict_diabetes_response(current_data)
            responses.append(prediction_result)
            current_data["waiting_for_more_questions"] = True
            current_data["current_question_index"] = 0
            current_data["current_symptoms"] = []
            return responses

        elif current_data.get("waiting_for_more_questions", False):
            if any(word in user_input_clean for word in positive_keywords):
                current_data["current_symptoms"].append(1)
            elif any(word in user_input_clean for word in negative_keywords):
                current_data["current_symptoms"].append(0)

            current_data["current_question_index"] += 1

            if current_data["current_question_index"] < len(questions):
                return questions[current_data["current_question_index"]]
            else:
                responses.append("سوالات به پایان رسید. در حال محاسبه پیش‌بینی دقیق‌تر...")

                # به‌روزرسانی لیست نهایی علائم
                current_data["symptoms"] = current_data["current_symptoms"]

                # انجام پیش‌بینی
                final_prediction_result = predict_more_accurate_diabetes_response(current_data)
                responses.append(final_prediction_result)

                # ذخیره داده در فایل بعد از کامل شدن پاسخ‌ها
                final_probability = predict_diabetes(
                    np.array([[
                        current_data["age"],
                        current_data["gender"]
                    ] + current_data["current_symptoms"] + [0] * (14 - len(current_data["current_symptoms"]))])
                )

                save_user_data_as_row(user_id, current_data, final_probability)

                # غیرفعال کردن حالت سوالات
                current_data["waiting_for_more_questions"] = False

                return responses

    if current_data["age"] is None:
        return "لطفاً سن خود را وارد کنید."
    if current_data["gender"] is None:
        return "لطفاً جنسیت خود را مشخص کنید (آقا یا خانم) و علائمتان را بگویید."

    return " ".join(responses)
# تابع پیش‌بینی و ارائه نتیجه  
def predict_diabetes_response(data):  
    age = data["age"]  
    gender = data["gender"]  
    symptoms = data["symptoms"]

    def has(symptom):
        return 1 if symptom in symptoms else 0

    input_features = np.array([[
        age,
        gender,
        has("پرادراری"),
        has("عطش"),
        has("کاهش وزن"),
        has("ضعف"),
        has("پرخوری"),
        has("عفونت قارچی"),
        has("تاری دید"),
        has("خارش"),
        has("عصبانیت"),
        has("تأخیر در بهبود"),
        has("فلج جزئی"),
        has("درد عضلانی"),
        has("ریزش مو"),
        has("چاقی")
    ]])

    print(f"ویژگی‌های ورودی برای مدل: {input_features}")
    probability = predict_diabetes(input_features)
    print(f"احتمال پیش‌بینی: {probability}")


    if probability > 50:
        return ("احتمال ابتلا به دیابت وجود دارد. اگر مایل باشد میتوانم با پرسش چند سوال احتمال ابتلا شما را به دیابت دقیق تر بررسی کنم "
                "(در صورت تمایل به پرسش سوالات کلمه 'سوال' را وارد کنید)")
    else:
        return (" احتمال ابتلا به دیابت پایین است. اگر مایل باشید میتوانم با پرسش چند سوال احتمال ابتلا شما را به دیابت دقیق تر بررسی کنم"
                "(در صورت تمایل به پرسش سوالات کلمه 'سوال' را وارد کنید)")

# در تابع جمع‌آوری اطلاعات  
def filter_nonnumeric(input_data):  
    # فقط مقادیر عددی را نگه می‌داریم  
    return [value for value in input_data if isinstance(value, (int, float))]  

def predict_more_accurate_diabetes_response(data):  
    age = data.get("age")  # سن  
    gender = data.get("gender")  # 0 برای خانم و 1 برای آقا  

    current_symptoms = data.get("current_symptoms", [])  # علائم  
    print("Current symptoms:", current_symptoms)  # چاپ برای بررسی  

    # ترکیب ویژگی‌ها  
    input_features = [age, gender] + current_symptoms  

    # فیلتر کردن علائم غیر عددی  
    filtered_input = filter_nonnumeric(input_features)  

    # بررسی تعداد ویژگی‌ها  
    expected_features_count = 16  # تعداد ویژگی‌های مورد انتظار  
    actual_features_count = len(filtered_input)  
    print(f"Expected features: {expected_features_count}, Actual features: {actual_features_count}")  

    # اگر تعداد ویژگی‌ها کمتر از حد انتظار باشد، باید ورودی را اصلاح کنید  
    if actual_features_count < expected_features_count:  
        # می‌توانید ویژگی‌های اضافی را به صورت صفر اضافه کنید  
        filtered_input += [0] * (expected_features_count - actual_features_count)  

    # تبدیل لیست به آرایه NumPy و تغییر شکل آن  
    input_array = np.array(filtered_input).reshape(1, -1)  
    # بررسی درستی ورودی برای مطمئن شدن از صحت آن  
    print("Input features for prediction:", input_array)  # چاپ برای بررسی  

    probability = predict_diabetes(input_array)  

    if probability > 50:  
        return ("<br>بر اساس پاسخ شما به سوالات، احتمال دیابت وجود دارد."
               "چند توصیه برای شما دارم :<br>لطفا در اولین فرصت با پزشک متخصص مشورت کنید<br>"
               "آزمایش‌های تشخیصی کامل‌تری نیاز است. ممکن است نیاز به درمان دارویی یا پیگیری مداوم داشته باشید.<br>"
               "رژیم غذایی خود را اصلاح کنید، مصرف قند، نمک و چربی را کاهش دهید.<br>"
               "ورزش منظم (حداقل ۳۰ دقیقه در روز، ۵ روز در هفته) را شروع یا حفظ کنید.<br>"
               "اگر سابقه خانوادگی بیماری دارید، مراقب علائم هشداردهنده باشید و به هیچ عنوان تغییرات مشکوک در وضعیت سلامتی را نادیده نگیرید.")
    else:  
        return ("بر اساس اطلاعات وارد شده، احتمال دیابت وجود ندارد.<br>وضعیت سلامتی شما خوب است"
                "<br>سبک زندگی سالم را حفظ کنید، مانند تغذیه متعادل و فعالیت بدنی منظم."
                "<br>به طور منظم چکاپ عمومی انجام دهید."
                "<br>مصرف دخانیات را به حداقل برسانید یا ترک کنید."
                "<br>استرس خود را مدیریت کنید و خواب کافی داشته باشید.")  

def save_user_data_as_row(user_id, data, probability):
    file_path = "diabetes_user_data.csv"
    file_exists = os.path.isfile(file_path)

    symptom_list = list(symptom_keywords.keys())
    fieldnames = ["user_id", "age", "gender"] + symptom_list + ["prediction"]

    with open(file_path, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            "user_id": user_id,
            "age": data.get("age"),
            "gender": data.get("gender")
        }

        # بررسی اینکه آیا علائم به صورت اسمی هستن یا فقط 0 و 1
        if all(isinstance(val, int) for val in data.get("symptoms", [])):
            # اگر علائم به صورت صفر و یک باشه (مثل بعد از سوالات)
            symptom_values = data["symptoms"]
            for i, symptom in enumerate(symptom_list):
                row[symptom] = symptom_values[i] if i < len(symptom_values) else 0
        else:
            # اگر علائم به صورت اسم باشن (مثل اول مکالمه)
            for symptom in symptom_list:
                row[symptom] = 1 if symptom in data.get("symptoms", []) else 0

        row["prediction"] = 1 if probability > 50 else 0

        writer.writerow(row)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


 
