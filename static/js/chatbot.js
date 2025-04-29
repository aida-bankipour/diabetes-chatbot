// Chatbot interaction functionality

// DOM Elements references
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const loadingIndicator = document.getElementById("loading");
let isDarkMode = false;
let isSpeaking = false;

// Add event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Send welcome message
    setTimeout(sendWelcomeMessage, 500);

    // Listen for Enter key on input field
    userInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") sendMessage();
    });

    // Set focus to input field
    userInput.focus();
});

// Type message with typing effect
function typeMessage(element, message) {
    let index = 0;
    const typingSpeed = 15; // milliseconds per character

    function typeChar() {
        if (index < message.length) {
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = message.slice(0, index + 1);
            element.innerHTML = tempDiv.innerHTML;
            index++;
            setTimeout(typeChar, typingSpeed);
        }
    }
    typeChar();
}

// Add message to chat (user or bot)
function addMessage(message, isBot) {
    const wrapper = document.createElement("div");
    wrapper.className = "max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow flex items-start gap-2";
    wrapper.style.animation = "fadeInChat 0.3s ease-out forwards";
    wrapper.classList.add("transition-all", "duration-300");

    if (isBot) {
        wrapper.classList.add("bg-indigo-100", "text-right", "self-start");
        wrapper.innerHTML = `
            
            <div class="flex-1"></div>
        `;
        const textDiv = document.createElement("div");
        wrapper.lastElementChild.appendChild(textDiv);
        chatBox.appendChild(wrapper);
        typeMessage(textDiv, message);
        speakText(message);
    } else {
        wrapper.classList.add("bg-indigo-600", "text-white", "text-left", "self-end");
        wrapper.innerHTML = `<div class='ml-2'>${message}`;
        chatBox.appendChild(wrapper);
    }

    // Apply dark mode if active
    if (isDarkMode && isBot) {
        wrapper.classList.remove("bg-indigo-100");
        wrapper.classList.add("bg-gray-700", "text-white");
    }

    // Scroll to bottom of chat
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Send welcome message
function sendWelcomeMessage() {
    const welcome = `سلام! 👋 به چت‌بات تشخیص دیابت خوش آمدید. 
    من می‌توانم به شما در مورد دیابت اطلاعاتی بدهم و احتمال ابتلای شما به دیابت را بر اساس علائم ارزیابی کنم. 
    میتوانید هر سوالی در رابطه با دیابت دارید از من بپرسید یا برای شروع ارزیابی، سن و جنسیت و علائم خود را وارد کنید .`;
    addMessage(welcome, true);
}

// Send message to server and get response
async function sendMessage() {
    if (!userInput.value.trim()) return;

    const message = userInput.value;
    const userId = document.getElementById("user-id").value;

    // Add user message to chat
    addMessage(message, false);

    // Clear input and show loading
    userInput.value = "";
    loadingIndicator.classList.remove("hidden");

    try {
        // Stop any current speech
        if (isSpeaking) {
            window.speechSynthesis.cancel();
        }

        // Send request to server
        const response = await fetch("/get_response", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: "message=" + encodeURIComponent(message) + "&user_id=" + encodeURIComponent(userId),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Add bot response to chat
        addMessage(data.response, true);
    }
    // catch (error) {
    //     console.error("Error getting response:", error);
    //     addMessage("خطا در ارتباط با سرور. لطفاً دوباره تلاش کنید.", true);
    // } 
    finally {
        loadingIndicator.classList.add("hidden");
        userInput.focus();
    }
}

// Voice input
function startListening() {
    if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
        addMessage("متأسفانه مرورگر شما از تشخیص صدا پشتیبانی نمی‌کند.", true);
        return;
    }

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "fa-IR";

    // Temporary message to show listening
    const tempMsg = document.createElement("div");
    tempMsg.className = "text-center text-sm text-indigo-600 py-2 animate-pulse";
    tempMsg.textContent = "🎤 در حال گوش دادن...";
    chatBox.appendChild(tempMsg);

    recognition.onresult = function (e) {
        if (chatBox.contains(tempMsg)) {
            chatBox.removeChild(tempMsg);
        }
        const transcript = e.results[0][0].transcript;
        userInput.value = transcript;
        sendMessage();
    };

    recognition.onerror = function (e) {
        if (chatBox.contains(tempMsg)) {
            chatBox.removeChild(tempMsg);
        }
        console.error("Speech recognition error:", e.error);

        let errorMessage = "خطا در تشخیص صدا";
        if (e.error === 'no-speech') {
            errorMessage = "صدایی شنیده نشد. لطفاً دوباره تلاش کنید.";
        } else if (e.error === 'network') {
            errorMessage = "خطای شبکه. لطفاً اتصال اینترنت خود را بررسی کنید.";
        } else if (e.error === 'not-allowed') {
            errorMessage = "دسترسی به میکروفن امکان‌پذیر نیست. لطفاً دسترسی را مجاز کنید.";
        }

        addMessage(errorMessage, true);
    };

    recognition.onend = function () {
        if (chatBox.contains(tempMsg)) {
            chatBox.removeChild(tempMsg);
        }
    };

    recognition.start();
}

// Voice output
function speakText(text) {
    if (!('speechSynthesis' in window)) {
        console.error("Browser doesn't support speech synthesis");
        return;
    }

    // Clean text from HTML tags and replace <br> with spaces
    const cleanText = text.replace(/<[^>]*>/g, "").replace(/<br>/g, ". ");

    // Create utterance with Persian language
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.lang = "fa-IR";
    utterance.rate = 0.9; // Slightly slower for better comprehension

    // Stop any current speech
    window.speechSynthesis.cancel();

    // Track speaking state
    isSpeaking = true;
    utterance.onend = function () {
        isSpeaking = false;
    };

    // Speak the text
    window.speechSynthesis.speak(utterance);
}

// Toggle dark mode
function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    document.getElementById("main-body").classList.toggle("dark-mode", isDarkMode);

    // Update chat messages for dark mode
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

// Simulate camera capture (since it's not fully implemented)
function simulateCapture() {
    addMessage("📸 قابلیت دوربین در نسخه نمایشی فعال نیست.", true);
}

// Simulate file upload (since it's not fully implemented)
function simulateFileUpload() {
    addMessage("📁 قابلیت آپلود فایل در نسخه نمایشی فعال نیست.", true);
}

// Generate a unique user ID if not exists
function generateUserId() {
    let userId = localStorage.getItem('diabetes_chatbot_user_id');
    if (!userId) {
        userId = 'user_' + Math.random().toString(36).substring(2, 15);
        localStorage.setItem('diabetes_chatbot_user_id', userId);
    }
    document.getElementById("user-id").value = userId;
}

// Quick responses for common questions
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

// Send a quick response
function sendQuickResponse(text) {
    userInput.value = text;
    sendMessage();
}

// Initialize when page loads
window.onload = function () {
    generateUserId();

    // Add quick response buttons after welcome message
    setTimeout(addQuickResponseButtons, 2000);
};
