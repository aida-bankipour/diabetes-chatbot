// Chatbot interaction functionality

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const loadingIndicator = document.getElementById("loading");
let isDarkMode = false;

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(sendWelcomeMessage, 500);

    userInput.addEventListener("keydown", e => {
        if (e.key === "Enter") sendMessage();
    });

    userInput.focus();
});

// Typing effect
function typeMessage(element, message) {
    return new Promise(resolve => {
        let index = 0;
        const typingSpeed = 15;

        function typeChar() {
            if (index < message.length) {
                const tempDiv = document.createElement("div");
                tempDiv.innerHTML = message.slice(0, index + 1);
                element.innerHTML = tempDiv.innerHTML;
                index++;
                setTimeout(typeChar, typingSpeed);
            } else {
                resolve();
            }
        }

        typeChar();
    });
}

// Add message (user or bot)
async function addMessage(message, isBot) {
    const wrapper = document.createElement("div");
    wrapper.className = "max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow flex items-start gap-2";
    wrapper.style.animation = "fadeInChat 0.3s ease-out forwards";
    wrapper.classList.add("transition-all", "duration-300");

    if (isBot) {
        wrapper.classList.add("bg-indigo-100", "text-right", "self-start");
        wrapper.innerHTML = `<div class="flex-1"></div>`;
        const textDiv = document.createElement("div");
        wrapper.lastElementChild.appendChild(textDiv);
        chatBox.appendChild(wrapper);
        await typeMessage(textDiv, message);
    } else {
        wrapper.classList.add("bg-indigo-600", "text-white", "text-left", "self-end");
        wrapper.innerHTML = `<div class='ml-2'>${message}`;
        chatBox.appendChild(wrapper);
    }

    if (isDarkMode && isBot) {
        wrapper.classList.remove("bg-indigo-100");
        wrapper.classList.add("bg-gray-700", "text-white");
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

// Welcome message
function sendWelcomeMessage() {
    const welcome = `سلام! 👋 به چت‌بات تشخیص دیابت خوش آمدید. 
    من می‌توانم به شما در مورد دیابت اطلاعاتی بدهم و احتمال ابتلای شما به دیابت را بر اساس علائم ارزیابی کنم. 
    میتوانید هر سوالی در رابطه با دیابت دارید از من بپرسید یا برای شروع ارزیابی، سن و جنسیت و علائم خود را وارد کنید .`;
    addMessage(welcome, true);
}

// Send message to server
async function sendMessage() {
    if (!userInput.value.trim()) return;

    const message = userInput.value;
    const userId = document.getElementById("user-id").value;

    addMessage(message, false);
    userInput.value = "";
    loadingIndicator.classList.remove("hidden");

    try {
        const response = await fetch("/get_response", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: "message=" + encodeURIComponent(message) + "&user_id=" + encodeURIComponent(userId),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        await addMessage(data.response, true);
    } finally {
        loadingIndicator.classList.add("hidden");
        userInput.focus();
    }
}

// Voice input (optional feature)
function startListening() {
    if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
        addMessage("متأسفانه مرورگر شما از تشخیص صدا پشتیبانی نمی‌کند.", true);
        return;
    }

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "fa-IR";

    const tempMsg = document.createElement("div");
    tempMsg.className = "text-center text-sm text-indigo-600 py-2 animate-pulse";
    tempMsg.textContent = "🎤 در حال گوش دادن...";
    chatBox.appendChild(tempMsg);

    recognition.onresult = e => {
        if (chatBox.contains(tempMsg)) {
            chatBox.removeChild(tempMsg);
        }
        const transcript = e.results[0][0].transcript;
        userInput.value = transcript;
        sendMessage();
    };

    recognition.onerror = e => {
        if (chatBox.contains(tempMsg)) {
            chatBox.removeChild(tempMsg);
        }

        let errorMessage = "خطا در تشخیص صدا";
        if (e.error === 'no-speech') errorMessage = "صدایی شنیده نشد. لطفاً دوباره تلاش کنید.";
        else if (e.error === 'network') errorMessage = "خطای شبکه. لطفاً اتصال اینترنت خود را بررسی کنید.";
        else if (e.error === 'not-allowed') errorMessage = "دسترسی به میکروفن امکان‌پذیر نیست. لطفاً دسترسی را مجاز کنید.";

        addMessage(errorMessage, true);
    };

    recognition.onend = () => {
        if (chatBox.contains(tempMsg)) {
            chatBox.removeChild(tempMsg);
        }
    };

    recognition.start();
}

// Dark mode toggle
function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    document.getElementById("main-body").classList.toggle("dark-mode", isDarkMode);

    const messages = chatBox.querySelectorAll("div.max-w-\\[75\\%\\]");
    messages.forEach(msg => {
        if (isDarkMode) {
            if (msg.classList.contains("bg-indigo-100")) {
                msg.classList.remove("bg-indigo-100");
                msg.classList.add("bg-gray-700", "text-white");
            }
        } else {
            if (msg.classList.contains("bg-gray-700")) {
                msg.classList.remove("bg-gray-700", "text-white");
                msg.classList.add("bg-indigo-100");
            }
        }
    });
}

// Simulated camera and file upload
function simulateCapture() {
    addMessage("📸 قابلیت دوربین در نسخه نمایشی فعال نیست.", true);
}

function simulateFileUpload() {
    addMessage("📁 قابلیت آپلود فایل در نسخه نمایشی فعال نیست.", true);
}

// Generate unique user ID
function generateUserId() {
    let userId = localStorage.getItem('diabetes_chatbot_user_id');
    if (!userId) {
        userId = 'user_' + Math.random().toString(36).substring(2, 15);
        localStorage.setItem('diabetes_chatbot_user_id', userId);
    }
    document.getElementById("user-id").value = userId;
}

// Quick replies
function addQuickResponseButtons() {
    const quickResponsesDiv = document.createElement('div');
    quickResponsesDiv.className = 'flex flex-wrap gap-2 justify-center my-3';
    quickResponsesDiv.innerHTML = `
        <button onclick="sendQuickResponse('دیابت چیست')" class="bg-indigo-100 hover:bg-indigo-200 text-indigo-800 px-3 py-1 rounded-full text-xs">دیابت چیست؟</button>
        <button onclick="sendQuickResponse('علائم دیابت چیست')" class="bg-indigo-100 hover:bg-indigo-200 text-indigo-800 px-3 py-1 rounded-full text-xs">علائم دیابت</button>
        <button onclick="sendQuickResponse('چگونه از دیابت پیشگیری کنیم')" class="bg-indigo-100 hover:bg-indigo-200 text-indigo-800 px-3 py-1 rounded-full text-xs">پیشگیری از دیابت</button>
    `;
    chatBox.appendChild(quickResponsesDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function sendQuickResponse(text) {
    userInput.value = text;
    sendMessage();
}

// Init
window.onload = () => {
    generateUserId();
    setTimeout(addQuickResponseButtons, 2000);
};
