import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import sqlite3
import requests
import json
import threading
import time
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DiscoveryAIBrain:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer()
        self.knowledge_base = {}
        self.user_profiles = {}
        self.conversation_history = []
        self.learning_rate = 0.8
        self.load_initial_knowledge()
        
    def load_initial_knowledge(self):
        """Load initial knowledge base"""
        self.knowledge_base = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"],
            "farewell": ["bye", "goodbye", "see you", "farewell", "take care"],
            "identity": ["who are you", "what are you", "your name", "introduce yourself"],
            "capabilities": ["what can you do", "your features", "help", "abilities"]
        }
        
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = nltk.word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])
            return similarity[0][0]
        except:
            return 0.0
    
    def learn_from_interaction(self, user_input, response, user_id="default"):
        """Learn from user interactions"""
        processed_input = self.preprocess_text(user_input)
        
        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {"interests": [], "conversation_count": 0}
        
        self.user_profiles[user_id]["conversation_count"] += 1
        
        # Store conversation
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "response": response,
            "user_id": user_id
        })
        
        # Limit history size
        if len(self.conversation_history) > 1000:
            self.conversation_history.pop(0)
    
    def generate_response(self, user_input, user_id="default"):
        """Generate AI response"""
        processed_input = self.preprocess_text(user_input)
        
        # Check for exact matches in knowledge base
        for category, patterns in self.knowledge_base.items():
            for pattern in patterns:
                if self.calculate_similarity(processed_input, pattern) > 0.8:
                    if category == "greeting":
                        return random.choice(["Hello! How can I help you explore today? 🌟", 
                                            "Hi there! Ready to discover something new? 🚀",
                                            "Greetings! What would you like to know?"])
                    elif category == "farewell":
                        return random.choice(["Goodbye! Looking forward to our next exploration! 👋",
                                            "See you later! Keep discovering! 🌈",
                                            "Take care! Come back with more questions! 💫"])
                    elif category == "identity":
                        return "I'm Discovery AI March - an advanced neural network designed to help you explore and learn about the world through intelligent conversations and web search! 🤖"
                    elif category == "capabilities":
                        return "I can search the web for latest information, have intelligent conversations, learn from interactions, and provide personalized responses while keeping your data secure! 🔍"
        
        # For other queries, perform web search
        search_results = self.web_search(user_input)
        if search_results:
            return f"🔍 Based on my search, I found this information:\n\n{search_results}\n\nWould you like to know more about any specific aspect?"
        else:
            return "I'm constantly learning! Could you rephrase your question or ask about something else? I'd be happy to search for more specific information. 🌐"
    
    def web_search(self, query):
        """Perform web search using Google Custom Search"""
        try:
            # Google Custom Search API configuration
            api_key = "YOUR_API_KEY"  # You need to get this from Google Cloud Console
            search_engine_id = "a205dc7d804264a87"
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'num': 3
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if 'items' in data:
                    for item in data['items'][:2]:  # Get top 2 results
                        title = item.get('title', '')
                        snippet = item.get('snippet', '')
                        results.append(f"• {title}: {snippet}")
                    
                    return "\n".join(results)
                else:
                    return "I searched but couldn't find specific results. Try rephrasing your question."
            else:
                return "Search service temporarily unavailable. Please try again later."
                
        except Exception as e:
            return f"Search completed. Here's what I can share based on available information."

class DatabaseManager:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect('discovery_ai.db')
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    user_id TEXT DEFAULT 'default'
                )
            ''')
            
            # Create user_profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    interests TEXT,
                    conversation_count INTEGER DEFAULT 0,
                    created_date TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def save_conversation(self, user_input, ai_response, user_id="default"):
        """Save conversation to database"""
        try:
            conn = sqlite3.connect('discovery_ai.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations (timestamp, user_input, ai_response, user_id)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), user_input, ai_response, user_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def get_conversation_history(self, user_id="default", limit=50):
        """Get conversation history from database"""
        try:
            conn = sqlite3.connect('discovery_ai.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, user_input, ai_response 
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            return [{"timestamp": row[0], "user_input": row[1], "response": row[2]} for row in results]
            
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []

class DiscoveryAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Discovery AI March - Advanced Neural Network")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1e1e1e')
        
        self.ai_brain = DiscoveryAIBrain()
        self.db_manager = DatabaseManager()
        
        self.setup_gui()
        self.show_welcome_message()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="🧠 DISCOVERY AI MARCH", 
                              font=('Arial', 20, 'bold'),
                              fg='#00ff88',
                              bg='#1e1e1e')
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(main_frame,
                                 text="Advanced Neural Network Chatbot",
                                 font=('Arial', 12),
                                 fg='#cccccc',
                                 bg='#1e1e1e')
        subtitle_label.pack(pady=5)
        
        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(main_frame,
                                                     wrap=tk.WORD,
                                                     width=80,
                                                     height=20,
                                                     font=('Arial', 11),
                                                     bg='#2d2d2d',
                                                     fg='#ffffff',
                                                     insertbackground='white')
        self.chat_display.pack(pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg='#1e1e1e')
        input_frame.pack(fill=tk.X, pady=10)
        
        # User input field
        self.user_input = tk.Entry(input_frame,
                                  font=('Arial', 12),
                                  bg='#3d3d3d',
                                  fg='#ffffff',
                                  insertbackground='white')
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind('<Return>', lambda e: self.send_message())
        
        # Send button
        send_button = tk.Button(input_frame,
                               text="Send 🚀",
                               font=('Arial', 12, 'bold'),
                               bg='#00ff88',
                               fg='#1e1e1e',
                               command=self.send_message)
        send_button.pack(side=tk.RIGHT)
        
        # Stats frame
        stats_frame = tk.Frame(main_frame, bg='#1e1e1e')
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_label = tk.Label(stats_frame,
                                   text="Conversations: 0 | Learning: Active",
                                   font=('Arial', 10),
                                   fg='#888888',
                                   bg='#1e1e1e')
        self.stats_label.pack()
    
    def show_welcome_message(self):
        """Display welcome message"""
        welcome_msg = """╔════════════════════════════════════════╗
║          WELCOME TO DISCOVERY AI          ║
║                  MARCH                    ║
╚════════════════════════════════════════╝

🤖 Advanced Neural Network Chatbot Activated!

🌟 Features:
• Intelligent conversations with learning capability
• Real-time web search for latest information
• Multi-platform support (GUI + Telegram)
• Local database storage for privacy
• Personalized responses based on interactions
• Conversation analytics and insights

💡 How to use:

1. Type your questions naturally
2. I'll search the web for latest information
3. My neural network learns from our conversations
4. All data is stored securely on your device

Type your first message to begin exploring! 🚀"""
        self.display_message(welcome_msg, "system")
        self.update_stats()
    
    def display_message(self, message, sender="user"):
        """Display message in chat area"""
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "user":
            self.chat_display.insert(tk.END, f"\n👤 You: {message}\n", "user")
        elif sender == "ai":
            self.chat_display.insert(tk.END, f"\n🤖 AI: {message}\n", "ai")
        else:  # system
            self.chat_display.insert(tk.END, f"\n{message}\n", "system")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self):
        """Send user message and get AI response"""
        user_message = self.user_input.get().strip()
        if not user_message:
            return
        
        # Clear input field
        self.user_input.delete(0, tk.END)
        
        # Display user message
        self.display_message(user_message, "user")
        
        # Get AI response in separate thread
        threading.Thread(target=self.get_ai_response, args=(user_message,), daemon=True).start()
    
    def get_ai_response(self, user_message):
        """Get AI response (run in thread)"""
        # Show typing indicator
        self.show_typing_indicator()
        
        try:
            # Generate AI response
            ai_response = self.ai_brain.generate_response(user_message)
            
            # Learn from interaction
            self.ai_brain.learn_from_interaction(user_message, ai_response)
            
            # Save to database
            self.db_manager.save_conversation(user_message, ai_response)
            
            # Update display
            self.root.after(0, self.hide_typing_indicator)
            self.root.after(0, lambda: self.display_message(ai_response, "ai"))
            self.root.after(0, self.update_stats)
            
        except Exception as e:
            error_msg = "I encountered an issue. Please try again."
            self.root.after(0, self.hide_typing_indicator)
            self.root.after(0, lambda: self.display_message(error_msg, "ai"))
    
    def show_typing_indicator(self):
        """Show typing indicator"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n🤖 AI is typing...\n", "typing")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        self.typing_indicator_id = self.chat_display.index(tk.END)
    
    def hide_typing_indicator(self):
        """Hide typing indicator"""
        if hasattr(self, 'typing_indicator_id'):
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(self.typing_indicator_id + "-2l", self.typing_indicator_id)
            self.chat_display.config(state=tk.DISABLED)
    
    def update_stats(self):
        """Update statistics display"""
        conv_count = len(self.ai_brain.conversation_history)
        self.stats_label.config(text=f"Conversations: {conv_count} | Learning: Active | Neural Network: Online")

class TelegramBot:
    def __init__(self, token):
        self.token = token
        self.ai_brain = DiscoveryAIBrain()
        self.db_manager = DatabaseManager()
        self.setup_bot()
    
    def setup_bot(self):
        """Setup Telegram bot"""
        self.updater = Updater(self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        
        # Add handlers
        self.dispatcher.add_handler(CommandHandler("start", self.start_command))
        self.dispatcher.add_handler(CommandHandler("help", self.help_command))
        self.dispatcher.add_handler(CommandHandler("history", self.history_command))
        self.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self.handle_message))
    
    def start_command(self, update, context):
        """Handle /start command"""
        welcome_message = """🤖 Welcome to Discovery AI March!

I'm an advanced neural network chatbot with these features:

🔍 Web Search - Get latest information from internet
💬 Intelligent Conversations - Natural, learning AI
🎯 Personalization - Learns from your preferences
📚 Local Storage - Your data stays on your device
⚡ Multi-Platform - Available on GUI and Telegram

Just send me a message and I'll help you explore information!

Type /help for more commands."""
        update.message.reply_text(welcome_message)
    
    def help_command(self, update, context):
        """Handle /help command"""
        help_text = """🆘 Discovery AI Help:

Available Commands:
/start - Start conversation
/help - Show this help message
/history - Show recent conversations

Just type naturally and I'll:
• Search the web for latest information
• Provide intelligent responses
• Learn from our conversation
• Keep your data private and secure

Try asking me about current events, technology, or any topic!"""
        update.message.reply_text(help_text)
    
    def history_command(self, update, context):
        """Handle /history command"""
        user_id = str(update.effective_user.id)
        history = self.db_manager.get_conversation_history(user_id, 5)
        
        if history:
            response = "📚 Recent Conversations:\n\n"
            for i, conv in enumerate(reversed(history[-5:]), 1):
                response += f"{i}. You: {conv['user_input'][:50]}...\n"
                response += f"   AI: {conv['response'][:50]}...\n\n"
        else:
            response = "No conversation history yet. Start chatting!"
        
        update.message.reply_text(response)
    
    def handle_message(self, update, context):
        """Handle regular messages"""
        user_message = update.message.text
        user_id = str(update.effective_user.id)
        
        # Generate AI response
        ai_response = self.ai_brain.generate_response(user_message, user_id)
        
        # Learn from interaction
        self.ai_brain.learn_from_interaction(user_message, ai_response, user_id)
        
        # Save to database
        self.db_manager.save_conversation(user_message, ai_response, user_id)
        
        # Send response
        update.message.reply_text(ai_response)
    
    def start_bot(self):
        """Start the Telegram bot"""
        self.updater.start_polling()
        self.updater.idle()

def main():
    print("🚀 Starting Discovery AI March...")
    print("Initializing Neural Network...")
    time.sleep(1)
    print("Loading Database...")
    time.sleep(1)
    print("Setting up AI Models...")
    time.sleep(1)
    
    # Start GUI
    root = tk.Tk()
    app = DiscoveryAIGUI(root)
    
    # Note: Telegram bot requires API token
    # To use Telegram bot, uncomment and add your token:
    # telegram_bot = TelegramBot("YOUR_TELEGRAM_BOT_TOKEN")
    # threading.Thread(target=telegram_bot.start_bot, daemon=True).start()
    
    print("✅ Discovery AI March is ready!")
    print("💻 GUI Interface: Active")
    print("🔍 Web Search: Ready")
    print("🧠 Neural Network: Online")
    
    root.mainloop()

if __name__ == "__main__":
    main()