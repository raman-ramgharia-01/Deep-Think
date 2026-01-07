document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatArea = document.getElementById('chatArea');
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeContainer = document.querySelector('.welcome-container');
    const themeToggle = document.getElementById('themeToggle');
    const newChatBtn = document.getElementById('newChatBtn');
    const clearAllBtn = document.getElementById('clearAllBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const settingsModal = document.getElementById('settingsModal');
    const closeSettings = document.getElementById('closeSettings');
    const themeSelect = document.getElementById('themeSelect');
    
    // Chat history
    let currentChatId = generateChatId();
    let chatHistory = [];
    
    // Initialize
    loadTheme();
    loadChatHistory();
    
    // Generate unique chat ID
    function generateChatId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
    
    // Theme functions
    function loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        themeSelect.value = savedTheme;
        applyTheme(savedTheme);
    }
    
    function applyTheme(theme) {
        if (theme === 'dark' || (theme === 'auto' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.body.classList.add('dark-mode');
            document.body.classList.remove('light-mode');
        } else {
            document.body.classList.add('light-mode');
            document.body.classList.remove('dark-mode');
        }
    }
    
    // Load chat history from localStorage
    function loadChatHistory() {
        const savedChats = JSON.parse(localStorage.getItem('deepThinkChats') || '[]');
        const chatList = document.getElementById('chatList');
        chatList.innerHTML = '';
        
        savedChats.forEach(chat => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            chatItem.innerHTML = `
                <i class="fas fa-comment"></i>
                <span class="chat-item-text">${chat.title || 'New Chat'}</span>
            `;
            
            chatItem.addEventListener('click', () => loadChat(chat.id));
            chatList.appendChild(chatItem);
        });
    }
    
    // Save chat to localStorage
    function saveChat() {
        const chats = JSON.parse(localStorage.getItem('deepThinkChats') || '[]');
        const existingChatIndex = chats.findIndex(chat => chat.id === currentChatId);
        
        const chatData = {
            id: currentChatId,
            title: chatHistory.length > 0 ? chatHistory[0].content.substring(0, 50) + '...' : 'New Chat',
            messages: chatHistory,
            timestamp: Date.now()
        };
        
        if (existingChatIndex > -1) {
            chats[existingChatIndex] = chatData;
        } else {
            chats.push(chatData);
        }
        
        localStorage.setItem('deepThinkChats', JSON.stringify(chats));
        loadChatHistory();
    }
    
    // Load a specific chat
    function loadChat(chatId) {
        const chats = JSON.parse(localStorage.getItem('deepThinkChats') || '[]');
        const chat = chats.find(c => c.id === chatId);
        
        if (chat) {
            currentChatId = chatId;
            chatHistory = chat.messages;
            
            // Hide welcome, show messages
            welcomeContainer.style.display = 'none';
            messagesContainer.style.display = 'block';
            
            // Clear and reload messages
            messagesContainer.innerHTML = '';
            chatHistory.forEach(msg => addMessageToUI(msg));
            
            // Update active state in sidebar
            document.querySelectorAll('.chat-item').forEach(item => {
                item.classList.remove('active');
            });
        }
    }
    
    // Add message to UI
    function addMessageToUI(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}-message`;
        
        const timestamp = new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        const avatarIcon = message.role === 'user' ? 'fas fa-user' : 'fas fa-robot';
        const senderName = message.role === 'user' ? 'You' : 'RamanTech';
        
        // Format code blocks in message content
        let formattedContent = message.content
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="$1">$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>');
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="message-sender">${senderName}</span>
                    <span class="message-time">${timestamp}</span>
                </div>
                <div class="message-text">${formattedContent}</div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Highlight code blocks
        if (window.hljs) {
            messageDiv.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }
    }
    
    // Send message
    async function sendMessage() {
        const message = messageInput.value.trim();
        
        if (!message) return;
        
        // Hide welcome message on first interaction
        if (welcomeContainer.style.display !== 'none') {
            welcomeContainer.style.display = 'none';
            messagesContainer.style.display = 'block';
        }
        
        // Add user message to UI
        const userMessage = {
            role: 'user',
            content: message,
            timestamp: Date.now()
        };
        
        addMessageToUI(userMessage);
        chatHistory.push(userMessage);
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;
        
        // Show loading
        loadingOverlay.style.display = 'flex';
        
        try {
            // Send to backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Add bot response
                const botMessage = {
                    role: 'bot',
                    content: data.response,
                    timestamp: Date.now()
                };
                
                addMessageToUI(botMessage);
                chatHistory.push(botMessage);
                
                // Save chat
                saveChat();
            } else {
                throw new Error(data.error || 'Request failed');
            }
        } catch (error) {
            console.error('Error:', error);
            
            const errorMessage = {
                role: 'bot',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: Date.now()
            };
            
            addMessageToUI(errorMessage);
            chatHistory.push(errorMessage);
        } finally {
            // Hide loading
            loadingOverlay.style.display = 'none';
            messageInput.focus();
        }
    }
    
    // New chat
    newChatBtn.addEventListener('click', function() {
        // Save current chat if it has messages
        if (chatHistory.length > 0) {
            saveChat();
        }
        
        // Reset for new chat
        currentChatId = generateChatId();
        chatHistory = [];
        
        // Show welcome message
        welcomeContainer.style.display = 'block';
        messagesContainer.style.display = 'none';
        messagesContainer.innerHTML = '';
        
        // Update active state in sidebar
        document.querySelectorAll('.chat-item').forEach(item => {
            item.classList.remove('active');
        });
    });
    
    // Clear all chats
    clearAllBtn.addEventListener('click', function() {
        if (confirm('Are you sure you want to clear all chats? This action cannot be undone.')) {
            localStorage.removeItem('deepThinkChats');
            newChatBtn.click();
            loadChatHistory();
        }
    });
    
    // Quick action cards
    document.querySelectorAll('.quick-action-card').forEach(card => {
        card.addEventListener('click', function() {
            const text = this.querySelector('span').textContent;
            messageInput.value = text;
            messageInput.focus();
            sendButton.disabled = false;
            autoResize(messageInput);
        });
    });
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendButton.disabled) {
                sendMessage();
            }
        }
    });
    
    messageInput.addEventListener('input', function() {
        sendButton.disabled = !this.value.trim();
        autoResize(this);
    });
    
    // Theme select change
    themeSelect.addEventListener('change', function() {
        localStorage.setItem('theme', this.value);
        applyTheme(this.value);
    });
    
    // Close settings modal
    closeSettings.addEventListener('click', function() {
        settingsModal.style.display = 'none';
    });
    
    // Close modal on outside click
    window.addEventListener('click', function(e) {
        if (e.target === settingsModal) {
            settingsModal.style.display = 'none';
        }
    });
    
    // Settings button
    themeToggle.addEventListener('click', function() {
        settingsModal.style.display = 'flex';
    });
    
    // Initialize textarea height
    messageInput.focus();
});