const chatBox = document.getElementById("chat-box");
const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const typingIndicator = document.getElementById("typing-indicator");
const suggestionsContainer = document.getElementById("suggestions-container");

function addMessage(text, sender, isTyping = false) {
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;
  
  if (isTyping) {
    msg.innerHTML = '';
    typeWriter(msg, text, 30);
  } else {
    msg.innerText = text;
  }
  
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addSuggestions(suggestions) {
  // Clear existing suggestions
  suggestionsContainer.innerHTML = '';
  
  // Add new suggestions as clickable buttons
  suggestions.forEach(suggestion => {
    const btn = document.createElement('button');
    btn.className = 'suggestion-btn';
    btn.innerText = suggestion;
    btn.onclick = () => {
      input.value = suggestion;
      sendMessage();
    };
    suggestionsContainer.appendChild(btn);
  });
}

function typeWriter(element, text, speed = 50) {
  let i = 0;
  function type() {
    if (i < text.length) {
      element.innerHTML += text.charAt(i);
      i++;
      setTimeout(type, speed);
    }
  }
  type();
}

function showTypingIndicator() {
  typingIndicator.style.display = 'flex';
  chatBox.appendChild(typingIndicator);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTypingIndicator() {
  typingIndicator.style.display = 'none';
  if (typingIndicator.parentNode) {
    typingIndicator.parentNode.removeChild(typingIndicator);
  }
}

function setLoading(loading) {
  sendBtn.disabled = loading;
  sendBtn.innerText = loading ? 'Sending...' : 'Send';
  input.disabled = loading;
}

async function sendMessage() {
  const message = input.value.trim();
  if (!message || sendBtn.disabled) return;

  addMessage(message, "user");
  input.value = "";
  setLoading(true);
  showTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ message })
    });

    hideTypingIndicator();

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    // Add main response with typing effect
    addMessage(data.reply, "bot", true);

    // Add suggestions to side panel
    if (data.suggestions && data.suggestions.length > 0) {
      setTimeout(() => {
        addSuggestions(data.suggestions);
      }, 1000);
    }

  } catch (error) {
    hideTypingIndicator();
    console.error('Error:', error);
    addMessage("❌ Sorry, I'm having trouble connecting to the server. Please try again.", "bot");
  } finally {
    setLoading(false);
  }
}

// Enter key support
input.addEventListener("keypress", function(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

// Auto-focus input
input.focus();

// Initialize with default suggestions
document.addEventListener('DOMContentLoaded', function() {
  const defaultSuggestions = [
    "What projects has Nathan worked on?",
    "What are Nathan’s top skills?",
    "What is Nathan background in computer science?",
    "How does Nathan approach problem solving?",
    "Why should we hire Nathan?"
  ];
  addSuggestions(defaultSuggestions);
});
