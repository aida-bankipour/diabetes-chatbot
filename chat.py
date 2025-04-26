from flask import Flask, render_template, request, jsonify  
import numpy as np  
import re  
import tensorflow as tf
tf.config.set_soft_device_placement(True)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)  
print("Checking files in mlp_model directory:")
print(os.listdir(os.path.join(BASE_DIR, 'mlp_model')))
# بارگذاری مدل  
model_path = os.path.join(BASE_DIR, 'mlp_model', 'mlp_model.keras')
print("Model path:", model_path)  # چاپ مسیر فایل مدل
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

# تابع پیش‌بینی  
def predict_diabetes(input_data):  
    prediction = model.predict(input_data)  
    probability = prediction[0][0] * 100  
    return probability  

@app.route("/")  
def home():  
    return render_template("index.html")  # نمایش صفحه HTML  

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

# تابع پردازش ورودی کاربر با اطلاعات کاربر  
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

    if "سلام" in user_input.strip():  
        return  "سلام به چت بات تشخیص دیابت خوش آمدید!<br>چطور می‌توانم به شما کمک کنم؟"  
    if "دیابت" in user_input.strip():  
        return "لطفاً سن و جنسیت و علائم خود را وارد کنید.<br> تا احتمال ابتلا شما به دیابت را بررسی کنم."  
    elif current_data.get("waiting_for_more_questions", False):  
        if user_input.strip() in ["سوال", "بپرس", "پرسش", "باشه"]:  
            current_data["current_question_index"] = 0  # شروع از سوال اول  
            return questions[current_data["current_question_index"]]
         
    if user_input.strip() in ["خداحافظ", "خدافظ", "خدانگهدار", "بای"]:  
            return "امیدوارم توانسته باشم به شما کمک کنم.<br> خدانگهدار."  

           
    # بررسی و ثبت اطلاعات کاربر  
    if current_data["gender"] is None:  
        if "خانم" in user_input:  
            current_data["gender"] = 0  # خانم 
        elif "زن" in user_input:  
            current_data["gender"] = 0
        elif "دختر" in user_input:  
            current_data["gender"] = 0     
        elif "آقا" in user_input:  
            current_data["gender"] = 1  # آقا 
        elif "مرد" in user_input:  
            current_data["gender"] = 1
        elif "پسر" in user_input:  
            current_data["gender"] = 1 
   

    # بررسی سن کاربر  
    age_match = re.search(r'(\d+)\s*سال', user_input)  # برای "34 سال"  
    if age_match and current_data["age"] is None:  
        current_data["age"] = int(age_match.group(1))  
        # response += f"\nسن شما ثبت شد: {current_data['age']} سال."  
    elif current_data["age"] is None:  
        # بررسی ورودی عددی به تنهایی  
        standalone_age_match = re.search(r'^\d+$', user_input.strip())  # برای "34"  
        if standalone_age_match:  
            current_data["age"] = int(standalone_age_match.group(0))  
            # response += f"\nسن شما ثبت شد: {current_data['age']} سال."  
        else:  
            return "لطفاً سن خود را وارد کنید."   

    # استخراج و نگهداری علائم  
    symptoms_in_input = []  
    for symptom in valid_symptoms:  
        if symptom in user_input and symptom not in current_data["symptoms"]:  
            symptoms_in_input.append(symptom)  

    if symptoms_in_input:  
        current_data["symptoms"].extend(symptoms_in_input)  

    # اگر سن و جنسیت مشخص شده باشند، فقط پیش‌بینی را انجام دهید.  
    if current_data["age"] is not None and current_data["gender"] is not None:  
        if not current_data.get("prediction_done", False):  
            current_data["prediction_done"] = True  
            prediction_result = predict_diabetes_response(current_data)  
            responses.append(prediction_result)  
            current_data["waiting_for_more_questions"] = True  # فعال کردن حالت انتظار  
            current_data["current_question_index"] = 0  # شروع از سوال اول  
            current_data["current_symptoms"] = []  # ایجاد لیست جدید برای ذخیره پاسخ‌ها  
            return responses  # ارسال اولین سوال بعد از پیش‌بینی  

        else:  
            if current_data.get("waiting_for_more_questions", False):  
                
                # بررسی پاسخ ورود به سوال قبلی  
                if user_input.lower() in "بله":  
                    current_data["current_symptoms"].append(1)  # ثبت پاسخ بله به عنوان 1  
                elif user_input.lower() in "دارم":  
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() in "اره":  
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() in "زیاد":  
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() in "تا حدودی":  
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() in "بیشتر مواقع":  
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() == "اکثرا":  
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() in "هستم":  
                    current_data["current_symptoms"].append(1)
                elif user_input.lower() == "میکنم":  
                    current_data["current_symptoms"].append(1)  # ثبت پاسخ بله به عنوان 1  
                elif user_input.lower() == "ندارم":  
                    current_data["current_symptoms"].append(0)
                elif user_input.lower() == "خیر":  
                    current_data["current_symptoms"].append(0)
                elif user_input.lower() == "نه":  
                    current_data["current_symptoms"].append(0)
                elif user_input.lower() == "کم":  
                    current_data["current_symptoms"].append(0)
                elif user_input.lower() == "نداشتم":  
                    current_data["current_symptoms"].append(0)
                elif user_input.lower() == "اصلا":  
                    current_data["current_symptoms"].append(0)
                elif user_input.lower() == "نمیکنم":  
                    current_data["current_symptoms"].append(0)  
                elif user_input.lower() == "نیستم":  
                    current_data["current_symptoms"].append(0)
                # افزایش اندیس سوالات  
                current_data["current_question_index"] += 1  

                # پرسش سوال جدید  
                if current_data["current_question_index"] < len(questions):  
                    return questions[current_data["current_question_index"]]  
                else:  
                    # همه سوالات به پایان رسیده است  
                    responses.append("سوالات همه به پایان رسیده است.<br> پیش‌بینی دقیق‌تری انجام می‌دهم....")  
                    # اضافه کردن علائم نهایی به current_data  
                    current_data["symptoms"] = current_data["current_symptoms"]  
                    final_prediction_result = predict_more_accurate_diabetes_response(current_data)  
                    responses.append(final_prediction_result)  
                    current_data["waiting_for_more_questions"] = False  # خاموش کردن حالت انتظار 
                    # send_delayed_responses(responses) 

    if current_data["age"] is None:  
        return "لطفاً سن خود را وارد کنید."  
    if current_data["gender"] is None:  
        return "لطفاً جنسیت خود را وارد کنید (خانم یا آقا)."  

    return " ".join(responses)  # بازگرداندن تمام پاسخ‌ها به صورت یکجا  
# تابع پیش‌بینی و ارائه نتیجه  
def predict_diabetes_response(data):  
    age = data["age"]  
    gender = data["gender"]  

    polyuria = 1 if ('ادرار بیش از حد معمول' in data["symptoms"] 
    or 'پرادراری' in data["symptoms"]
    or 'ادرار زیاد' in data["symptoms"]) else 0  
    
    polydipsia = 1 if( 'عطش' in data["symptoms"]
    or 'تشنگی' in data["symptoms"]) else 0  
    
    sudden_weight_loss = 1 if ('کاهش وزن' in data["symptoms"]
    or 'کاهش ناگهانی وزن' in data["symptoms"]
    or 'افت وزن' in data["symptoms"]
    or 'افت شدید وزن' in data["symptoms"]) else 0  
    
    weakness = 1 if( 'ضعف' in data["symptoms"]
    or 'بی حالی' in data["symptoms"]) else 0  
    
    polyphagia = 1 if 'پرخوری' in data["symptoms"] else 0  
    genital_thrush = 1 if 'عفونت قارچی' in data["symptoms"] else 0  
    visual_blurring = 1 if ('تاری دید' in data["symptoms"]
    or 'کاهش میدان دید' in data["symptoms"]) else 0  
    
    itching = 1 if ('خارش' in data["symptoms"] 
    or 'خشکی' in data["symptoms"]) else 0  
    
    irritability = 1 if ('تحریک‌ پذیری' in data["symptoms"]
    or 'عصبانیت' in data["symptoms"]
    or 'عصبی' in data["symptoms"]) else 0  
    
    delayed_healing = 1 if ('تأخیر در بهبود' in data["symptoms"]
    or 'زخم هام دیر خوب میشوند' in data["symptoms"]
    or 'آثار زخم باقی میماند' in data["symptoms"]) else 0  
    
    partial_paresis = 1 if ('فلج جزئی' in data["symptoms"]
    or 'درد عضلانی' in data["symptoms"]
    or 'کشیدگی' in data["symptoms"]) else 0  
    
    muscle_stiffness = 1 if( 'سفتی عضلات' in data["symptoms"]
    or 'درد' in data["symptoms"]) else 0  
    alopecia = 1 if 'ریزش مو' in data["symptoms"] else 0  
    obesity = 1 if 'چاقی' in data["symptoms"] else 0  

    # ساخت آرایه ویژگی‌ها برای مدل  
    input_features = np.array([[age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia,  
                                genital_thrush, visual_blurring, itching, irritability, delayed_healing,  
                                partial_paresis, muscle_stiffness, alopecia, obesity]])  

    print(f"ویژگی‌های ورودی برای مدل: {input_features}")  
    probability = predict_diabetes(input_features)  
    print(f"احتمال پیش‌بینی: {probability}")  
    
    if probability > 50:  
        return ("🔮پیش‌بینی: بر اساس علائمی که وارد کردیداحتمال ابتلا به دیابت وجود دارد .<br>آیا مایل هستید چند سوال بپرسم و پیش بینی دقیق تری از وضعیت ابتلا شما به دیابت انجام دهم؟<br> (اگر مایل هستید کلمه سوال را وارد کنید)")
        
    else:  
        return ("🔮پیش‌بینی: بر اساس علائمی که وارد کردیداحتمال ابتلا به دیابت وجود ندارد .<br>آیا مایل هستید چند سوال بپرسم و پیش بینی دقیق تری از وضعیت ابتلا شما به دیابت انجام دهم؟<br> (اگر مایل هستید کلمه سوال را وارد کنید)")
        
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
        return ("<br>بر اساس اطلاعات وارد شده، احتمال دیابت وجود دارد.<br>لطفا در اولین فرصت با پزشک متخصص مشورت کنید<br>و با انجام آزمایشات تخصصی وضعیت سلامت خود را بررسی کنید.")
          
    else:  
        return ("بر اساس اطلاعات وارد شده، احتمال دیابت وجود ندارد.<br>وضعیت سلامتی شما خوب است")  

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
 
